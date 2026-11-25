#!/usr/bin/env python3
"""Read-only B1 cctv-3.0 video-reader verification. No GPU or model needed.

Tests actual PTS and RGB pixels against an independent full PyAV decode,
checks frame/track export timestamps, and performs negative error-code checks.
The OpenCV comparison is diagnostic, not a strict binary pixel requirement.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import yaml


def code_of(exc):
    for key in ("code", "error_code"):
        value = getattr(exc, key, None)
        if value is not None:
            return str(value)
    return str(exc)


def test_error(fn, expected, label):
    try:
        fn()
    except Exception as exc:
        if expected not in code_of(exc) and expected not in str(exc):
            raise AssertionError(f"{label}: expected {expected}, got {type(exc).__name__}: {exc}") from exc
        print(f"PASS {label}: {expected}")
    else:
        raise AssertionError(f"{label}: expected {expected} but succeeded")


def reference_frames(path, targets, offset, tolerance):
    """Independent full sequential PyAV scan; selects nearest PTS with earlier tie break."""
    best = {t: None for t in targets}
    first_pts = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                pts = float(frame.pts * frame.time_base)
                if first_pts is None:
                    first_pts = pts
                ms = offset + round((pts - first_pts) * 1000)
                source = "pts"
            else:
                rate = stream.average_rate
                if rate is None or float(rate) <= 0:
                    raise RuntimeError("No PTS and no FPS")
                ms = offset + round(idx * 1000 / float(rate))
                source = "estimated"
            for t in targets:
                dist = abs(ms - t)
                old = best[t]
                if old is None or (dist, ms) < (old["distance"], old["timestamp_ms"]):
                    best[t] = {"distance": dist, "timestamp_ms": ms,
                               "frame_index": idx, "time_source": source,
                               "rgb": frame.to_ndarray(format="rgb24")}
    for t, rec in best.items():
        if rec is None or rec["distance"] > tolerance:
            raise AssertionError(f"Reference frame not found for {t}ms")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, default=Path("."))
    p.add_argument("--video-path", type=Path, default=Path("data/CityFlow-NL/CityFlow_selected/c033/vdo.avi"))
    p.add_argument("--video-id", default="cityflow_S05_c033")
    p.add_argument("--manifest", type=Path, default=Path("data/manifest.jsonl"))
    p.add_argument("--config", type=Path, default=Path("configs/vision.yaml"))
    p.add_argument("--frames-jsonl", type=Path, default=Path("runs/b_litedetr_dev_001/b3_b4_t3_candidate/frames.jsonl"))
    p.add_argument("--output", type=Path, default=Path("runs/b_litedetr_dev_001/b1_validation_report.json"))
    p.add_argument("--samples", default="0,200,75600,151200,340600")
    args = p.parse_args()
    root = args.project.resolve()
    resolve = lambda x: x if x.is_absolute() else root / x
    path, manifest, config, frame_file, out = map(resolve, [args.video_path, args.manifest, args.config, args.frames_jsonl, args.output])
    for f in (path, manifest, config, frame_file):
        if not f.is_file():
            raise FileNotFoundError(f)
    sys.path.insert(0, str(root / "src"))
    import b_video_io
    manifests = [json.loads(s) for s in manifest.read_text(encoding="utf-8").splitlines() if s.strip()]
    manifest_rec = next((x for x in manifests if x["video_id"] == args.video_id), None)
    if manifest_rec is None:
        raise ValueError(f"video_id not found in manifest: {args.video_id}")
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    tolerance = int(cfg["vision"]["max_frame_error_ms"])
    start = int(manifest_rec["available_start_ms"])
    end = int(manifest_rec["available_end_ms"])
    offset = int(manifest_rec.get("source_offset_ms") or 0)
    targets = sorted(set(int(s.strip()) for s in args.samples.split(",") if s.strip()))
    for t in targets:
        if not start <= t < end:
            raise ValueError(f"Sample {t} outside manifest [{start},{end})")
    print("B1 validation video:", path)
    print("Manifest window:", start, end, "max_error:", tolerance)
    reader = b_video_io.configure_reader(manifest, config, path_overrides={args.video_id: str(path)})
    rows = [json.loads(s) for s in frame_file.read_text(encoding="utf-8").splitlines() if s.strip()]
    frame_by_req = {int(r["requested_timestamp_ms"]): r for r in rows if r.get("video_id") == args.video_id}
    for t in targets:
        if t not in frame_by_req:
            raise ValueError(f"No exported frames.jsonl row for {t}ms")
    expected = reference_frames(path, targets, offset, tolerance)
    detailed = []
    for t in targets:
        got = b_video_io.read_frame(args.video_id, t)
        ref = expected[t]
        for key in ("frame_index", "time_source"):
            if got[key] != ref[key]:
                raise AssertionError(f"{t}ms {key}: reader {got[key]} vs independent {ref[key]}")
        if got["actual_timestamp_ms"] != ref["timestamp_ms"]:
            raise AssertionError(f"{t}ms PTS mismatch")
        if got["rgb"].dtype != np.uint8 or got["rgb"].ndim != 3 or got["rgb"].shape[-1] != 3:
            raise AssertionError(f"{t}ms invalid RGB array")
        if not np.array_equal(got["rgb"], ref["rgb"]):
            raise AssertionError(f"{t}ms PyAV RGB pixel mismatch")
        exported = frame_by_req[t]
        if exported["timestamp_ms"] != got["actual_timestamp_ms"] or exported["frame_index"] != got["frame_index"]:
            raise AssertionError(f"{t}ms JSONL/reader PTS or frame index mismatch")
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ref["frame_index"]))
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise AssertionError(f"{t}ms OpenCV cannot decode frame {ref['frame_index']}")
        cv_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if cv_rgb.shape != got["rgb"].shape:
            raise AssertionError(f"{t}ms OpenCV/PyAV shape mismatch")
        diff = np.abs(cv_rgb.astype(np.int16) - got["rgb"].astype(np.int16))
        row = {"requested_timestamp_ms": t, "frame_index": got["frame_index"],
               "actual_timestamp_ms": got["actual_timestamp_ms"], "time_source": got["time_source"],
               "rgb_sha256": hashlib.sha256(got["rgb"].tobytes()).hexdigest(),
               "opencv_pixel_exact": bool(np.array_equal(cv_rgb, got["rgb"])),
               "opencv_mean_abs_diff": round(float(diff.mean()), 6),
               "opencv_max_abs_diff": int(diff.max())}
        detailed.append(row)
        print(f"PASS PTS/RGB/JSONL {t}ms -> idx {row['frame_index']} PTS {row['actual_timestamp_ms']}; "
              f"OpenCV mean diff {row['opencv_mean_abs_diff']}")
    test_error(lambda: b_video_io.read_frame("__B1_UNKNOWN_VIDEO__", 0), "VIDEO_NOT_FOUND", "unknown video")
    test_error(lambda: b_video_io.read_frame(args.video_id, start - 1), "OUT_OF_RANGE", "before start")
    test_error(lambda: b_video_io.read_frame(args.video_id, end), "OUT_OF_RANGE", "end exclusive")
    # Broken path: use a dedicated reader so the genuine configured instance remains unchanged.
    bad_reader = b_video_io.CCTVFrameReader(manifest, config, path_overrides={args.video_id: str(root / "__missing_b1_video__.avi")})
    test_error(lambda: bad_reader.read_frame(args.video_id, targets[0]), "DECODE_FAILED", "missing video file")
    report = {"test": "B1 real video PTS/RGB and negative errors", "video_id": args.video_id,
              "pass": True, "sample_count": len(targets), "max_frame_error_ms": tolerance,
              "notes": ["Reference PyAV PTS/RGB independently scanned in one pass",
                        "OpenCV pixel differences are diagnostic only; quality metrics may differ",
                        "Synthetic VFR/single-frame and damaged-media fixtures are not tested here",
                        "Other videos and track-state reset are not tested here"],
              "samples": detailed,
              "errors_tested": ["VIDEO_NOT_FOUND", "OUT_OF_RANGE", "DECODE_FAILED (missing file)"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {out}")
    with out.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print("PASS B1 core test. Report:", out)
    print("PENDING: synthetic VFR/single-frame, corrupted bytes, 3-video validation, manual bbox review")

if __name__ == "__main__":
    main()
