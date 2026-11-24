"""Adapt frame-state pixel search to the live web result and media contract."""

import numpy as np

from embedding.contracts import read_jsonl
from integrations.f_metadata import image_exclusions
from .frame_state import cached_windows, rank_windows


def search(catalog, allowed, vector, args, store, *, full, limit):
    encoder = store.encoder
    batch = store.config["batch_size_gpu" if encoder.device.startswith("cuda") else "batch_size_cpu"]
    windows, embeddings = [], []
    for key, (engine, media) in catalog.items():
        for video in read_jsonl(engine.root.parent / "manifest.jsonl"):
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
            rows, vectors, folder = cached_windows(path, video, exclusions, 30, False,
                                                   store.root / "pixel-cache", encoder, batch, pixel_attention=True)
            for row in rows:
                row.update(thumbnail_url=f"/media/pixel/{folder.name}/{row['frame_indices'][1]}/thumbnail",
                           video_url=f"/media/{key}/{first['vector_row']}/video", display_name=name,
                           width=video["width"], height=video["height"],
                           source_offset_ms=video["source_offset_ms"], source_sha256=video["file_sha256"])
            windows.extend(rows)
            embeddings.append(vectors)
    if not windows:
        return [], 0, 0

    vectors = np.concatenate(embeddings)
    ranked = rank_windows(windows, vectors, vector[0], pixel_attention=True)[:limit]
    results = []
    for rank, row in enumerate(ranked, 1):
        result = {k: row[k] for k in ("video_id", "start_ms", "end_ms", "source_offset_ms", "source_sha256",
                                      "display_name", "width", "height", "thumbnail_url", "video_url")}
        results.append(dict(result, rank=rank, representative_ms=row["timestamp_ms"],
                            frame_index=row["frame_indices"][1], playback_time_ms=row["timestamp_ms"] - row["source_offset_ms"],
                            score=row["similarity"], ranking_score=row["similarity"],
                            representation="scene", track_uid=None, bbox=None,
                            event_ids=[], reasons=[row["source"]]))
    return results, len(windows), vectors.size // encoder.dim
