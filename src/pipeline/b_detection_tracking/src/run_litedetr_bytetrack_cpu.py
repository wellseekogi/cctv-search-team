#!/usr/bin/env python3
"""Run Ultralytics ByteTrack on saved Lite-DETR PTS detections (CPU)."""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


def read_records(path):
    records = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Non-object record at line {line_no}")
            records.append(obj)
    if not records:
        raise ValueError(f"No detections in {path}")
    expected_vid = records[0]["video_id"]
    for idx, rec in enumerate(records):
        if rec.get("video_id") != expected_vid:
            raise ValueError(f"Mixed video IDs in record {idx}")
        if rec.get("requested_timestamp_ms") != idx * 200:
            raise ValueError(f"Noncontiguous 200ms sampling in record {idx}")
        if abs(rec["actual_timestamp_ms"] - rec["requested_timestamp_ms"]) > 100:
            raise ValueError(f"PTS outside tolerance in record {idx}")
    return records


def tracker_args(fuse):
    d = YAML.load(check_yaml("bytetrack.yaml"))
    d.update(track_high_thresh=0.25, track_low_thresh=0.10,
             new_track_thresh=0.25, track_buffer=30, match_thresh=0.8,
             fuse_score=fuse)
    return IterableSimpleNamespace(**d)


def input_boxes(dets, h, w):
    rows = []
    for d in dets:
        b = [float(x) for x in d["bbox"]]
        score = float(d["confidence"])
        label = int(d["class_id"])
        if not (np.all(np.isfinite(b)) and np.isfinite(score)):
            continue
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        rows.append([*b, score, label])
    array = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return Boxes(torch.from_numpy(array), orig_shape=(h, w)).cpu().numpy()


def run_one(records, output, *, min_conf, fuse):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    trk = BYTETracker(args=tracker_args(fuse))
    total_obs = 0
    unique_ids = set()
    zero_frames = 0
    input_dets = 0
    with output.open("w", encoding="utf-8") as fh:
        for rec in records:
            dets = [d for d in rec["detections"] if float(d["confidence"]) >= min_conf]
            input_dets += len(dets)
            tracks = trk.update(input_boxes(dets, int(rec["height"]), int(rec["width"])))
            if len(tracks) == 0:
                zero_frames += 1
            objects = []
            for t in tracks:
                if len(t) < 7:
                    raise ValueError(f"Unexpected ByteTrack output row: {t}")
                x1, y1, x2, y2, tid, score, cls = t[:7]
                uid = f"det01:{rec['video_id']}:{int(tid)}"
                unique_ids.add(uid)
                objects.append({
                    "track_uid": uid, "local_track_id": str(int(tid)),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(score), "class_id": int(cls),
                })
            total_obs += len(objects)
            fh.write(json.dumps({
                "record_type": "tracking_experiment",
                "video_id": rec["video_id"],
                "frame_index": int(rec["frame_index"]),
                "timestamp_ms": int(rec["actual_timestamp_ms"]),
                "requested_timestamp_ms": int(rec["requested_timestamp_ms"]),
                "tracks": objects,
            }, ensure_ascii=False) + "\n")
    return dict(frames=len(records), track_observations=total_obs,
                unique_track_ids=len(unique_ids), frames_with_zero_tracks=zero_frames,
                avg_input_detections=round(input_dets / len(records), 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("runs/b_litedetr_dev_001/detections_dev_pts.jsonl"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/b_litedetr_dev_001/cpu_tracking"))
    args = p.parse_args()
    records = read_records(args.input)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {args.output_dir}. Choose a new --output-dir.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, conf, fused in [("B_iou_025", .25, False), ("C_iou_010", .10, False)]:
        dest = args.output_dir / f"tracking_{name}.jsonl"
        row = run_one(records, dest, min_conf=conf, fuse=fused)
        row["experiment"] = name
        summary.append(row)
        print(name, row)
    with (args.output_dir / "tracking_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["experiment", "frames", "track_observations", "unique_track_ids", "frames_with_zero_tracks", "avg_input_detections"])
        writer.writeheader()
        writer.writerows(summary)
    print("PASS: CPU tracking experiments complete", args.output_dir)


if __name__ == "__main__":
    main()
