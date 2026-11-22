"""Index object-independent CCTV intervals and search their visual evidence."""

from collections import defaultdict
from pathlib import Path
import tempfile
import time

import av
import numpy as np
from filelock import FileLock

from embedding.contracts import check_vectors, digest, file_sha256, read_jsonl, write_json, write_jsonl
from integrations.f_metadata import image_exclusions
from sampling.frame_state import propose_windows
from team_search.baseline import source_video


def interval_rows(path, video, exclusions, budget, uniform):
    offset = video["source_offset_ms"]
    windows = propose_windows(
        path, budget=budget, uniform=uniform, time_source=video.get("time_source", "pts"),
        start_sec=(video["available_start_ms"] - offset) / 1000,
        end_sec=(video["available_end_ms"] - offset) / 1000,
        excluded=tuple(((row["start_ms"] - offset) / 1000, (row["end_ms"] - offset) / 1000)
                       for row in exclusions),
    )
    rows = []
    for window in windows:
        times = [offset + round(t * 1000) for t in window.frame_times_sec]
        rows.append(dict(video_id=video["video_id"], start_ms=offset + round(window.start_sec * 1000),
                         end_ms=offset + round(window.end_sec * 1000), timestamp_ms=times[1],
                         frame_times_ms=times, frame_indices=list(window.frame_indices),
                         roi_xyxy=list(window.roi_xyxy), activity_score=window.score, source=window.source))
    return rows


def encode_windows(path, rows, encoder, batch_size):
    """Decode each requested frame once, keeping the three times and both views separate."""
    needed = defaultdict(list)
    for i, row in enumerate(rows):
        for t, index in enumerate(row["frame_indices"]):
            needed[index].append((i, t))
    vectors = np.empty((len(rows), 3, 2, encoder.dim), dtype=np.float32)
    if not needed:
        return vectors
    images, slots = [], []

    def flush():
        for slot, vector in zip(slots, encoder.encode_image(images), strict=True):
            vectors[slot] = vector
        images.clear()
        slots.clear()

    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in needed:
                continue
            image = frame.to_image()
            width, height = image.size
            for i, t in needed.pop(index):
                x, y, right, bottom = rows[i]["roi_xyxy"]
                box = (int(x * width), int(y * height), int(np.ceil(right * width)), int(np.ceil(bottom * height)))
                for view, rgb in enumerate((image, image.crop(box))):
                    images.append(rgb)
                    slots.append((i, t, view))
                    if len(images) >= batch_size:
                        flush()
            if not needed:
                break
    if needed:
        raise ValueError(f"Selected frame missing in {path}: {min(needed)}")
    if images:
        flush()
    return vectors


def cached_windows(path, video, exclusions, budget, uniform, cache, encoder, batch_size):
    key = digest([video, exclusions, budget, uniform, encoder.metadata, file_sha256(Path(__file__)),
                  file_sha256(Path(__file__).resolve().parents[1] / "sampling/frame_state.py")])
    cache.mkdir(parents=True, exist_ok=True)
    folder = cache / key
    with FileLock(str(folder) + ".lock"):
        if not folder.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temp:
                stage = Path(temp) / "index"
                stage.mkdir()
                started = time.perf_counter()
                rows = interval_rows(path, video, exclusions, budget, uniform)
                proposal_seconds = time.perf_counter() - started
                vectors = encode_windows(path, rows, encoder, batch_size)
                write_jsonl(stage / "windows.jsonl", rows)
                np.save(stage / "vectors.npy", vectors, allow_pickle=False)
                write_json(stage / "cost.json", dict(proposal_seconds=proposal_seconds,
                           embedding_seconds=time.perf_counter() - started - proposal_seconds,
                           image_encodes=len(rows) * 6, budget=budget, policy="uniform" if uniform else "state",
                           camera=video["camera_id"], sha256=video["file_sha256"]))
                stage.rename(folder)
    rows = read_jsonl(folder / "windows.jsonl")
    vectors = np.load(folder / "vectors.npy", allow_pickle=False)
    if vectors.shape != (len(rows), 3, 2, encoder.dim):
        raise ValueError(f"Invalid frame-state cache: {folder}")
    check_vectors(vectors.reshape(-1, encoder.dim), len(rows) * 6, encoder.dim)
    return rows, vectors, folder


def gallery(metadata, data_root, cache, datasets, split, rate, encoder, batch_size, uniform=False):
    if rate <= 0:
        raise ValueError("windows-per-minute must be positive")
    rows, vectors, folders = [], [], []
    for dataset in datasets:
        queries = read_jsonl(metadata / dataset / "queries.jsonl")
        for video in read_jsonl(metadata / dataset / "manifest.jsonl"):
            if split != "all" and video["split"] != split:
                continue
            path = source_video(metadata, data_root, video)
            exclusions = image_exclusions(queries, dict(video, f_video_id=video["video_id"]))
            duration = video["available_end_ms"] - video["available_start_ms"]
            budget = max(1, round(duration * rate / 60000))
            part, embedding, folder = cached_windows(path, video, exclusions, budget, uniform, cache,
                                                   encoder, batch_size)
            rows.extend(part)
            vectors.append(embedding)
            folders.append(folder)
    if not vectors:
        raise ValueError("No video matches the requested dataset and split")
    return rows, np.concatenate(vectors), folders


def rank_windows(rows, vectors, query, *, center_only=False):
    # Equal weight for the full scene and class-free local view. Activity never labels an action.
    scores = vectors @ query
    scores = (scores[:, 1] if center_only else scores.mean(axis=1)).mean(axis=1)
    order = np.argsort(-scores, kind="stable")
    return [dict(rows[i], similarity=float(scores[i])) for i in order]


def search(args, config, encoder):
    datasets = ("meva", "virat") if args.dataset == "all" else (args.dataset,)
    batch = config["batch_size_gpu" if encoder.device.startswith("cuda") else "batch_size_cpu"]
    rows, vectors, _ = gallery(args.metadata_root, args.data_root, args.state_cache, datasets,
                               args.split, args.windows_per_minute, encoder, batch)
    ranked = rank_windows(rows, vectors, encoder.encode_text([args.question])[0])
    for i, row in enumerate(ranked[:args.top_k], 1):
        print(f"{i}. {row['video_id']} | {row['start_ms']/1000:.2f}–{row['end_ms']/1000:.2f}초"
              f" | 유사도 {row['similarity']:.4f} | {row['source']}")
    if not ranked:
        print("검색 가능한 구간이 없습니다.")
