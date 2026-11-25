"""Embed one B camera export in bounded batches, preserving observation metadata."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from embedding.config import create_encoder, load_config
from embedding.contracts import SCHEMA, file_sha256, write_json, write_jsonl
from embedding.track_input import SequentialVideoReader, iter_embedding_batches, iter_jsonl


def run(args, encoder=None):
    config = load_config(args.config)
    videos = [row for row in iter_jsonl(args.manifest) if row.get("video_id") == args.video_id]
    if len(videos) != 1:
        raise ValueError("Manifest must contain exactly one matching video_id")
    video = videos[0]
    # Original pixels must match the file used for B's coordinates and timestamps.
    if file_sha256(args.video) != video["file_sha256"]:
        raise ValueError("Original video SHA256 differs from B manifest")
    batch_size = args.batch_size
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be positive")
    if not 0 <= args.min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    if type(args.shard_size) is not int or args.shard_size < 1:
        raise ValueError("shard_size must be positive")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Use a new output directory")
    if encoder is None:
        encoder = create_encoder(config)
    if batch_size is None:
        batch_size = config["batch_size_gpu"] if encoder.device.startswith("cuda") else config["batch_size_cpu"]
    output.mkdir(parents=True)
    count, parts = 0, 0
    pending_items, pending_vectors = [], []

    def publish(items, vectors, part_index):
        with tempfile.TemporaryDirectory(prefix=".part-", dir=output) as temp:
            stage = Path(temp)
            write_jsonl(stage / "items.jsonl", [{**item, "vector_row": i} for i, item in enumerate(items)])
            np.save(stage / "vectors.npy", vectors, allow_pickle=False)
            # Publish metadata and vectors together; a partial shard is never visible.
            os.replace(stage, output / f"part-{part_index:06d}")

    tracks = iter_jsonl(args.tracks)
    try:
        with SequentialVideoReader(args.video, video, iter_jsonl(args.frames)) as reader:
            batches = iter_embedding_batches(tracks, reader.read_frame, encoder, batch_size=batch_size,
                                             margin=config["crop_margin_ratio"], min_confidence=args.min_confidence)
            for items, vectors in batches:
                pending_items.extend(items)
                pending_vectors.append(vectors)
                if len(pending_items) >= args.shard_size:
                    # Shards end on model batch boundaries, bounding memory by shard_size + batch_size.
                    publish(pending_items, np.concatenate(pending_vectors), parts)
                    count += len(pending_items)
                    parts += 1
                    pending_items, pending_vectors = [], []
            if pending_items:
                publish(pending_items, np.concatenate(pending_vectors), parts)
                count += len(pending_items)
                parts += 1
        result = {"schema_version": SCHEMA, "format": "track-embedding-shards-v1", "status": "success",
                  "video_id": args.video_id, "split": video["split"], "row_count": count, "parts": parts,
                  "embedding_dim": encoder.dim, "dtype": "float32", "normalization": "l2",
                  "encoder": encoder.metadata, "crop_margin_ratio": config["crop_margin_ratio"],
                  "batch_size": batch_size, "shard_size": args.shard_size, "min_confidence": args.min_confidence,
                  "source_video_sha256": video["file_sha256"]}
        write_json(output / "manifest.json", result)
        return result
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "completed_rows": count,
                                           "completed_parts": parts, "message": str(exc)})
        raise
    finally:
        tracks.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("tracks", "frames", "manifest", "video", "video-id", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--config", default="configs/embedding/embedding.siglip2.yaml")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--min-confidence", type=float, default=0)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False))
        return 0
    except Exception as exc:
        parser.exit(1, f"{type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
