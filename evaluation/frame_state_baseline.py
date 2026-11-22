"""Build or score the original B → A hybrid → C gallery on the MEVA eval split."""
import argparse
import json
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from embedding import create_encoder
from embedding.contracts import digest, file_sha256, read_jsonl, write_json
from filelock import FileLock
from rag.engine import FrameRecommender
from team_search.__main__ import get_config
from team_search.baseline import ensure_observations, source_video
from team_search.full import build_video

ROOT = Path(__file__).resolve().parents[1]


def build_gallery(videos, queries, args, config, encoder):
    settings = json.loads((ROOT / "configs/person/hybrid.yaml").read_text(encoding="utf8"))
    indexes = []
    args.cache.mkdir(parents=True, exist_ok=True)
    for video in videos:
        path = source_video(ROOT / "data", args.data_root, video)
        folder = ensure_observations(video, path, args.cache, args.b_device or encoder.device)
        inputs = [file_sha256(folder / name) for name in ("frames.jsonl", "tracks.jsonl", "validation_report.json")]
        run = args.cache / ("a-" + digest([video, inputs, settings, queries, encoder.metadata,
                                          config["crop_margin_ratio"], file_sha256(ROOT / "sampling/core.py"),
                                          file_sha256(ROOT / "team_search/full.py")]))
        with FileLock(str(run) + ".lock"):
            if not run.exists():
                with tempfile.TemporaryDirectory(dir=args.cache) as temp:
                    stage = Path(temp) / "gallery"
                    stage.mkdir()
                    build_video(folder, video, args.data_root, queries, settings, config, encoder, stage)
                    stage.rename(run)
        indexes.append(run / "index")
    return indexes


def score_indexes(indexes, videos, queries, truths, korean, encoder, horizon_ms):
    engines = [FrameRecommender(path, encoder) for path in indexes]
    sources = {video["video_id"]: video["file_sha256"] for video in videos}
    fps = {video["video_id"]: video["fps"] for video in videos}
    actual_sources = {}
    candidates = []
    by_camera = defaultdict(list)
    camera_counts = defaultdict(Counter)
    counts = Counter()
    for engine in engines:
        for name, sha in engine.index["sources"].items():
            video_id = name.removeprefix("meva_")
            if video_id in actual_sources:
                raise ValueError("Duplicate baseline source: " + video_id)
            actual_sources[video_id] = sha
        rows = [r for r in engine.items if horizon_ms is None or r["actual_timestamp_ms"] < horizon_ms]
        candidates.append(rows)
        for row in rows:
            camera = row["video_id"].removeprefix("meva_")
            by_camera[camera].append(row["actual_timestamp_ms"])
            camera_counts[camera][row["representation"]] += 1
            counts[row["representation"]] += 1
    if actual_sources != sources:
        raise ValueError("Baseline sources do not match the MEVA eval split")
    stamps = {k: sorted(set(v)) for k, v in by_camera.items()}
    # Match 40 distinct retrieval positions per camera, retaining native crop/scene scores.
    matched = {}
    if set(stamps) == set(sources) and all(len(times) >= 40 for times in stamps.values()):
        matched = {k: {v[(2 * j + 1) * len(v) // 80] for j in range(40)} for k, v in stamps.items()}

    rows = []
    for q in queries:
        if q["split"] != "eval" or q["modality"] != "text":
            continue
        qid = q["query_id"]
        positive = [r for r in truths if r["query_id"] == qid and r["relevance"] > 0]
        if not positive:
            raise ValueError("Missing positives: " + qid)
        for language, prompt in (("ko", korean[qid]), ("en", q["text"])):
            vector = encoder.encode_text([prompt])
            hits = [h for engine, pool in zip(engines, candidates)
                    for h in engine.ranked_frames(vector, pool, len(pool))]
            hits.sort(key=lambda h: (-h["ranking_score"], h["video_id"], h["frame_index"], h["item_id"]))
            for rank, hit in enumerate(hits, 1):
                hit["rank"] = rank

            def first_match(ranked, use_f_clock=False):
                for position, hit in enumerate(ranked, 1):
                    video_id = hit["video_id"].removeprefix("meva_")
                    stamp = round(hit["frame_index"] * 1000 / fps[video_id]) if use_f_clock else hit["timestamp_ms"]
                    if any(video_id == r["video_id"] and r["start_ms"] <= stamp < r["end_ms"] for r in positive):
                        return position, hit
                return None, None

            rank, match = first_match(hits)
            f_rank, _ = first_match(hits, use_f_clock=True)
            subset = [h for h in hits if matched and h["timestamp_ms"] in matched[h["video_id"].removeprefix("meva_")]]
            matched_rank, _ = first_match(subset)
            rows.append({"query_id": qid, "language": language, "kind": q["kind"],
                         "rank": rank, "first_correct": match, "rank_f_cfr": f_rank,
                         "rank_matched_40": matched_rank, "top5": hits[:5]})
    return {"horizon_ms": horizon_ms, "items": sum(counts.values()), "representations": dict(counts),
            "items_by_camera": dict(camera_counts), "unique_frames": sum(map(len, stamps.values())),
            "timestamps_ms": stamps,
            "matched_40": {k: {"timestamps_ms": sorted(s), "frames": len(s),
                               "items": sum(t in s for t in by_camera[k])} for k, s in matched.items()},
            "indexes": [e.index_hash for e in engines], "source_sha256": sources, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, action="append", help="Existing baseline index; repeat per camera")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/state-baseline")
    parser.add_argument("--data-root", type=Path, default=ROOT / "runs/source-data")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--embedding-config", type=Path, default=ROOT / "configs/embedding.siglip2.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--b-device", choices=("cpu", "cuda"))
    parser.add_argument("--horizon-ms", type=int, help="Score only candidates before this exclusive timestamp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon_ms is not None and args.horizon_ms <= 0:
        parser.error("--horizon-ms must be positive")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    data = ROOT / "data/meva"
    videos = [v for v in read_jsonl(data / "manifest.jsonl") if v["split"] == "eval"]
    queries = read_jsonl(data / "queries.jsonl")
    korean = json.loads((ROOT / "evaluation/meva_ko.json").read_text(encoding="utf8"))
    config = get_config(args)
    encoder = create_encoder(config)
    start = time.perf_counter()
    indexes = args.index or build_gallery(videos, queries, args, config, encoder)
    ready = time.perf_counter()
    report = score_indexes(indexes, videos, queries, read_jsonl(data / "qrels.jsonl"),
                           korean, encoder, args.horizon_ms)
    report["timing_s"] = {"prepare_or_reuse": ready - start, "score": time.perf_counter() - ready}
    report["embedding"] = []
    for index in indexes:
        status = json.loads((index / "embedding_status.json").read_text(encoding="utf8"))
        report["embedding"].append({k: status[k] for k in (
            "requested", "succeeded", "failed", "read_crop_ms", "encode_success_ms", "wall_ms")})
    if not args.index:
        report["b"] = {}
        for video in videos:
            folder = args.cache / "baseline_T3" / video["file_sha256"]
            detections = read_jsonl(folder / "detections_pts.jsonl")
            handoff = json.loads((folder / "handoff/validation_report.json").read_text(encoding="utf8"))
            report["b"][video["video_id"]] = {
                "sampled_frames": handoff["sampled_frames"], "tracks_observations": handoff["tracks_observations"],
                "model_forward_s": sum(row["inference_seconds"] for row in detections),
                "max_forward_s": max(row["inference_seconds"] for row in detections),
                "max_abs_pts_error_ms": max(abs(row["actual_timestamp_ms"] - row["requested_timestamp_ms"])
                                            for row in detections),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(f"{len(report['rows'])} query/language ranks, {report['items']} items, "
          f"{report['unique_frames']} distinct frames: {args.output}", flush=True)


if __name__ == "__main__":
    main()
