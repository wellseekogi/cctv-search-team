"""Adapt frame-state pixel search to the live web result and media contract."""

from io import BytesIO

import av
import numpy as np

from embedding.contracts import read_jsonl
from integrations.f_metadata import image_exclusions
from .frame_state import cached_windows, rank_windows


def search(catalog, allowed, vector, args, store, *, full, limit):
    encoder = store.encoder
    batch = store.config["batch_size_gpu" if encoder.device.startswith("cuda") else "batch_size_cpu"]
    windows, embeddings = [], []
    for key, (engine, media) in catalog.items():
        for slot, video in enumerate(read_jsonl(engine.root.parent / "manifest.jsonl")):
            vid = video["video_id"]
            if allowed and vid not in allowed:
                continue
            first = next((r for r in engine.items if r["video_id"] == vid), None)
            if first is None:
                continue
            path, _, name = media[vid]
            exclusions = []
            if full:
                queries = read_jsonl(args.metadata_root / video["dataset"] / "queries.jsonl")
                exclusions = image_exclusions(queries, video)
            rows, vectors, _ = cached_windows(path, video, exclusions, 30, False,
                                              store.root / "pixel-cache", encoder, batch, pixel_attention=True)
            for row in rows:
                row.update(thumbnail_url=f"/media/pixel/{key}/{slot}/{row['frame_indices'][1]}",
                           video_url=f"/media/{key}/{first['vector_row']}/video", display_name=name,
                           width=video["width"], height=video["height"],
                           source_offset_ms=video["source_offset_ms"], source_sha256=video["file_sha256"])
            windows.extend(rows)
            embeddings.append(vectors)
    if not windows:
        return [], 0

    ranked = rank_windows(windows, np.concatenate(embeddings), vector[0], pixel_attention=True)[:limit]
    results = []
    for rank, row in enumerate(ranked, 1):
        result = {k: row[k] for k in ("video_id", "start_ms", "end_ms", "source_offset_ms", "source_sha256",
                                      "display_name", "width", "height", "thumbnail_url", "video_url")}
        results.append(dict(result, rank=rank, representative_ms=row["timestamp_ms"],
                            frame_index=row["frame_indices"][1], playback_time_ms=row["timestamp_ms"] - row["source_offset_ms"],
                            score=row["similarity"], ranking_score=row["similarity"],
                            representation="scene", track_uid=None, bbox=None,
                            event_ids=[], reasons=[row["source"]]))
    return results, len(windows)


def thumbnail(shard, slot, frame_index):
    engine, media = shard
    videos = read_jsonl(engine.root.parent / "manifest.jsonl")
    if not 0 <= slot < len(videos) or frame_index < 0:
        return None
    path = media[videos[slot]["video_id"]][0]
    with av.open(str(path)) as video:
        for index, frame in enumerate(video.decode(video=0)):
            if index == frame_index:
                image = frame.to_image()
                image.thumbnail((320, 180))
                output = BytesIO()
                image.save(output, format="PNG")
                return output.getvalue()
    return None
