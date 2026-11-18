#!/usr/bin/env python3
"""B1 synthetic single-frame, VFR, corrupt-video fixture tests.

Read-only with respect to project data. Generates test inputs in TemporaryDirectory.
Uses the project's actual src/b_video_io.py implementation.
"""
import argparse
import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import yaml


def error_code(exc):
    return str(getattr(exc, "code", getattr(exc, "error_code", "")))


def expect_error(callback, expected, label):
    try:
        callback()
    except Exception as exc:
        code = error_code(exc)
        if code != expected and expected not in str(exc):
            raise AssertionError(f"{label}: expected {expected}, got {type(exc).__name__}: {code}: {exc}") from exc
        print(f"PASS {label}: {expected}")
        return
    raise AssertionError(f"{label}: expected {expected}; call succeeded")


def make_video(path, timestamps):
    # Lossy encoding is fine: RGB is compared to RGB decoded from the SAME file.
    # Time base in milliseconds makes variable frame presentation times explicit.
    with av.open(str(path), mode="w", format="matroska") as out:
        stream = out.add_stream("mpeg4", rate=10)
        stream.width = 128
        stream.height = 96
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for idx, ms in enumerate(timestamps):
            a = np.zeros((96, 128, 3), dtype=np.uint8)
            a[:, :, 0] = (31 + idx * 49) % 256
            a[:, :, 1] = (100 + idx * 27) % 256
            a[:, :, 2] = (200 - idx * 37) % 256
            a[12:40, 20 + idx * 4:52 + idx * 4] = [240, 30, 90]
            frame = av.VideoFrame.from_ndarray(a, format="rgb24")
            frame.pts = ms
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)


def read_reference(path):
    frames = []
    with av.open(str(path)) as inp:
        stream = inp.streams.video[0]
        first = None
        for i, frame in enumerate(inp.decode(stream)):
            assert frame.pts is not None and frame.time_base is not None, "Fixture PTS absent"
            pts = float(frame.pts * frame.time_base)
            if first is None:
                first = pts
            frames.append((i, round((pts-first)*1000), frame.to_ndarray(format="rgb24")))
    assert frames, f"No decoded frames from {path}"
    return frames


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=Path("."))
    ap.add_argument("--config", type=Path, default=Path("configs/vision.yaml"))
    ap.add_argument("--output", type=Path, default=Path("runs/b_litedetr_dev_001/b1_fixture_report.json"))
    args = ap.parse_args()
    project = args.project.resolve()
    cfg = args.config if args.config.is_absolute() else project / args.config
    report_path = args.output if args.output.is_absolute() else project / args.output
    if not cfg.is_file():
        raise FileNotFoundError(cfg)
    sys.path.insert(0, str(project / "src"))
    import b_video_io
    config = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    config.setdefault("vision", {})["max_frame_error_ms"] = 100
    report = {"suite": "B1 synthetic fixtures", "passed": False, "tests": [],
              "warning": "Synthetic fixtures do NOT count as three distinct real CCTV videos"}

    with tempfile.TemporaryDirectory(prefix="b1_cctv_fixtures_") as temp:
        root = Path(temp)
        single_path = root / "single.mkv"
        vfr_path = root / "vfr.mkv"
        corrupt_path = root / "corrupted.mkv"
        make_video(single_path, [0])
        make_video(vfr_path, [0, 100, 300, 520])
        corrupt_path.write_bytes(b"NOT_A_VIDEO\x00" * 512)
        single = read_reference(single_path)
        vfr = read_reference(vfr_path)
        print("Single-frame decoded PTS:", [r[1] for r in single])
        print("VFR decoded PTS:", [r[1] for r in vfr])
        assert [r[1] for r in vfr] == [0, 100, 300, 520], "Fixture PTS changed in encoding: cannot test VFR"
        assert len(single) == 1 and single[0][1] == 0

        manifests = [
            {"video_id": "b1_fixture_single", "available_start_ms": 0,
             "available_end_ms": 1000, "source_offset_ms": 0},
            {"video_id": "b1_fixture_vfr", "available_start_ms": 0,
             "available_end_ms": 650, "source_offset_ms": 0},
            {"video_id": "b1_fixture_corrupt", "available_start_ms": 0,
             "available_end_ms": 1000, "source_offset_ms": 0},
        ]
        manifest_file = root / "manifest.jsonl"
        manifest_file.write_text("".join(json.dumps(x) + "\n" for x in manifests), encoding="utf-8")
        config_file = root / "vision.yaml"
        config_file.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        overrides = {"b1_fixture_single": str(single_path), "b1_fixture_vfr": str(vfr_path),
                     "b1_fixture_corrupt": str(corrupt_path)}
        reader = b_video_io.CCTVFrameReader(manifest_file, config_file, path_overrides=overrides)

        checks = [
            ("single exact", "b1_fixture_single", 0, single[0]),
            ("single within tolerance", "b1_fixture_single", 90, single[0]),
            ("VFR tie earlier", "b1_fixture_vfr", 200, vfr[1]),
            ("VFR nearest", "b1_fixture_vfr", 270, vfr[2]),
            ("VFR final", "b1_fixture_vfr", 520, vfr[3]),
            ("VFR near final", "b1_fixture_vfr", 590, vfr[3]),
        ]
        for label, vid, target, expected in checks:
            got = reader.read_frame(vid, target)
            idx, pts, rgb = expected
            assert got["frame_index"] == idx, (label, got["frame_index"], idx)
            assert got["actual_timestamp_ms"] == pts, (label, got["actual_timestamp_ms"], pts)
            assert got["time_source"] == "pts", (label, got["time_source"])
            assert got["rgb"].shape == rgb.shape and np.array_equal(got["rgb"], rgb), label
            print(f"PASS {label}: request {target}ms -> idx={idx}, PTS={pts}ms, RGB exact")
            report["tests"].append({"name": label, "requested_ms": target, "actual_ms": pts,
                                    "frame_index": idx, "status": "PASS"})

        # Missing frame within a VALID manifest window (> 100ms from only frame).
        negatives = [
            ("single no nearby frame", lambda: reader.read_frame("b1_fixture_single", 150), "FRAME_NOT_FOUND"),
            ("VFR no nearby frame", lambda: reader.read_frame("b1_fixture_vfr", 410), "FRAME_NOT_FOUND"),
            ("single end exclusive", lambda: reader.read_frame("b1_fixture_single", 1000), "OUT_OF_RANGE"),
            ("VFR end exclusive", lambda: reader.read_frame("b1_fixture_vfr", 650), "OUT_OF_RANGE"),
            ("corrupted bytes", lambda: reader.read_frame("b1_fixture_corrupt", 0), "DECODE_FAILED"),
        ]
        for label, cb, code in negatives:
            expect_error(cb, code, label)
            report["tests"].append({"name": label, "status": "PASS", "expected_error": code})

        # Switching source IDs on same reader must return the right source, not stale pixels.
        a = reader.read_frame("b1_fixture_single", 0)
        b = reader.read_frame("b1_fixture_vfr", 300)
        c = reader.read_frame("b1_fixture_single", 0)
        assert np.array_equal(a["rgb"], c["rgb"])
        assert not np.array_equal(a["rgb"], b["rgb"])
        print("PASS reader video switch: no stale RGB")
        report["tests"].append({"name": "reader video switch", "status": "PASS"})

    report["passed"] = True
    report["count"] = len(report["tests"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        raise FileExistsError(f"Will not overwrite: {report_path}")
    with report_path.open("x", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"PASS B1 synthetic fixture suite ({report['count']} checks). Report: {report_path}")
    print("PENDING: 3 distinct real CCTV videos, ByteTrack state reset, manual bbox review")


if __name__ == "__main__":
    run()
