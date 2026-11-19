"""Run with python -m sampling --input fixtures/event_trace --out runs/event_trace."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .core import InputError, run_sampling


def read_jsonl(path):
    rows = []
    for line, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        try:
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError("expected an object")
            rows.append(row)
        except ValueError as exc:
            raise InputError(f"{path.name}:{line}: {exc}") from exc
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                            for row in rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="A 담당 객체 Event 정책 실행")
    parser.add_argument("--input", type=Path, default=Path("fixtures/event_trace"))
    parser.add_argument("--config", type=Path, default=Path("configs/sampling.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs/event_trace"))
    parser.add_argument("--run-id", default="event-trace")
    parser.add_argument("--frame-reader", help="motion: module:function returning B/C frame dictionary")
    args = parser.parse_args()
    # Never silently overwrite an earlier run or its evidence.
    if args.out.exists() and any(args.out.iterdir()):
        print(f"Output folder is not empty; choose a new --out: {args.out}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        # JSON is a YAML 1.2 subset. This starter config needs no PyYAML installation.
        config = json.loads(args.config.read_text(encoding="utf-8-sig"))
        names = ("manifest.jsonl", "frames.jsonl", "tracks.jsonl", "video_status.jsonl", "query_exclusions.jsonl")
        inputs = [read_jsonl(args.input / name) for name in names]
        loader = None
        if config["sampling"]["policy"] == "motion":
            import numpy as np
            from PIL import Image
            def loader(frame):
                relative = frame.get("rgb_path")
                if not isinstance(relative, str):
                    raise InputError("motion CLI requires rgb_path on every readable frame; use frame_loader for B integration")
                path = (args.input / relative).resolve()
                if not path.is_relative_to(args.input.resolve()):
                    raise InputError("rgb_path must stay inside the input directory")
                with Image.open(path) as image:
                    return np.asarray(image.convert("RGB"))
            if args.frame_reader:
                import importlib
                from integrations.b_reader import motion_frame
                module, function = args.frame_reader.split(":", 1)
                reader = getattr(importlib.import_module(module), function)
                def loader(frame):
                    return motion_frame(reader, frame)
        events, samples, summary = run_sampling(config, *inputs, run_id=args.run_id, frame_loader=loader)
        motion_scores = summary.pop("motion_scores", None)
        summary["input_sha256"] = {name: hashlib.sha256((args.input / name).read_bytes()).hexdigest() for name in names}
        write_jsonl(args.out / "events.jsonl", events)
        write_jsonl(args.out / "samples.jsonl", samples)
        if motion_scores is not None:
            write_jsonl(args.out / "motion_scores.jsonl", motion_scores)
        write_json(args.out / "selection_summary.json", summary)
        write_jsonl(args.out / "errors.jsonl", [])
        (args.out / "sampling.yaml").write_bytes(args.config.read_bytes())
        write_json(args.out / "stage_status.json", {"stage": "sampling", "status": "success", "source_run_id": args.run_id})
        print(json.dumps(summary["counts"], ensure_ascii=False))
        return 0
    except (InputError, ValueError, KeyError, TypeError, OSError) as exc:
        write_jsonl(args.out / "errors.jsonl", [{"stage": "sampling", "code": "INVALID_INPUT", "message": str(exc)}])
        write_json(args.out / "stage_status.json", {"stage": "sampling", "status": "failed", "source_run_id": args.run_id})
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        write_jsonl(args.out / "errors.jsonl", [{"stage": "sampling", "code": "INTERNAL_ERROR", "message": str(exc)}])
        write_json(args.out / "stage_status.json", {"stage": "sampling", "status": "failed", "source_run_id": args.run_id})
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
