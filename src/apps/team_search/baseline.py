"""Prepare the published Lite-DETR/ByteTrack B handoff when none was supplied."""

import argparse
import csv
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from urllib.request import urlopen

from filelock import FileLock

from embedding.contracts import file_sha256, within

ROOT = Path(__file__).resolve().parents[3]
REVISION = "6392353c1f37ece35844d6578faffe307c9e737d"
WEIGHTS = "r50_s2ex3_49.9.pth"
WEIGHTS_SHA = "90c2399406ea4e2c83fb76ed1182f9d9120f80639f1fab5d0909425247af94a3"


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        if path.exists():
            if file_sha256(path) != expected:
                raise ValueError(f"SHA256 불일치: {path}")
            return
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            pending = Path(temp) / path.name
            with urlopen(url, timeout=120) as response, pending.open("wb") as stream:
                shutil.copyfileobj(response, stream, 1024 * 1024)
            if file_sha256(pending) != expected:
                raise ValueError(f"다운로드 SHA256 불일치: {path.name}")
            pending.rename(path)


def source_video(metadata, root, video):
    path = within(root, video["relative_path"])
    if path.is_file():
        if file_sha256(path) != video["file_sha256"]:
            raise ValueError(f"원본 영상 해시 불일치: {path}")
        return path
    inventory = metadata / video["dataset"] / "data_inventory.csv"
    with inventory.open(encoding="utf-8-sig", newline="") as stream:
        matches = [r for r in csv.DictReader(stream) if r["file_sha256"] == video["file_sha256"]]
    if len(matches) != 1:
        raise ValueError(f"원본 다운로드 주소를 확인하세요: {path}")
    download(matches[0]["source_url"], path, video["file_sha256"])
    return path


def prepare_model(root):
    repo = root / "Lite-DETR"
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "checkout.lock")):
        if not repo.exists():
            with tempfile.TemporaryDirectory(dir=root) as temp:
                stage = Path(temp) / "repo"
                subprocess.run(
                    ["git", "clone", "--quiet", "https://github.com/IDEA-Research/Lite-DETR.git", str(stage)],
                    check=True,
                )
                subprocess.run(["git", "-C", str(stage), "checkout", "--quiet", REVISION], check=True)
                # The upstream package imports the optional CUDA extension even in native PyTorch mode.
                module = stage / "models/dino/ops/functions/ms_deform_attn_func.py"
                source = module.read_text(encoding="utf-8").replace(
                    "\nimport MultiScaleDeformableAttention as MSDA\n",
                    "\ntry:\n    import MultiScaleDeformableAttention as MSDA\nexcept ImportError:\n    MSDA = None\n",
                )
                module.write_text(source, encoding="utf-8")
                stage.rename(repo)
    download(
        f"https://github.com/IDEA-Research/Lite-DETR/releases/download/litedetr/{WEIGHTS}",
        root / WEIGHTS,
        WEIGHTS_SHA,
    )
    return repo


def export(video, source, destination, work, model_dir, device):
    import torch
    from unittest.mock import patch

    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))

    sys.path.insert(0, str(ROOT / "src/pipeline/b_detection_tracking/src"))
    from build_b3_b4_generic_person import run as export_tracks
    from run_generic_bytetrack_t3 import run as track
    from litedetr_inference import run_one_video

    repo = prepare_model(model_dir)
    sys.path.insert(0, str(repo))
    from models.dino import build_dino
    from models.dino.ops.modules.ms_deform_attn import MSDeformAttn

    transforms = SimpleNamespace(**runpy.run_path(str(repo / "datasets/transforms.py")))
    settings = {}
    for name in ("coco_transformer.py", "DINO_4scale.py"):
        settings.update(runpy.run_path(str(repo / "config/DINO" / name)))
    settings.update(
        device=device, num_expansion=3, enc_scale=1, key_aware=True, debug=False, dataset_file="coco"
    )
    with patch("models.dino.backbone.is_main_process", return_value=False):
        model, _, post = build_dino(SimpleNamespace(**settings))
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(model_dir / WEIGHTS, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state["model"])
    del state
    for module in model.modules():
        if isinstance(module, MSDeformAttn):
            module.use_pytorch_version = True
    model.to(device).eval()
    preprocess = transforms.Compose(
        [
            transforms.RandomResize([480], max_size=854),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    times = list(range(video["available_start_ms"], video["available_end_ms"], 200))
    dets = work / "detections_pts.jsonl"
    run_one_video(
        dict(
            video_id=video["video_id"],
            path=source,
            output=dets,
            timestamps=times,
            total=len(times),
            offset_ms=video["source_offset_ms"],
            width=video["width"],
            height=video["height"],
        ),
        model,
        post,
        preprocess,
    )
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        tracking = track(dets, Path(tmp) / "tracks", len(times))
        export_tracks(
            SimpleNamespace(
                project=ROOT,
                detection_path=dets.resolve(),
                video_path=source.resolve(),
                tracking_path=Path(tracking["tracking_jsonl"]).resolve(),
                output_dir=destination.resolve(),
                video_id=video["video_id"],
                expected_frames=len(times),
                source_id="litedetr01",
            )
        )


def ensure_observations(video, source, cache, device):
    work = cache / "baseline_T3" / video["file_sha256"]
    work.mkdir(parents=True, exist_ok=True)
    output = work / "handoff"
    with FileLock(str(work / "prepare.lock")):
        if not output.is_dir():
            print(f"B Lite-DETR · ByteTrack: {video['video_id']}", flush=True)
            model_dir = Path(os.getenv("CCTV_B_MODEL_DIR", cache / "b-model")).resolve()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "team_search.baseline",
                    json.dumps(video),
                    str(source.resolve()),
                    str(output.resolve()),
                    str(work.resolve()),
                    str(model_dir),
                    device,
                ],
                cwd=ROOT,
                check=True,
            )
    return output


if __name__ == "__main__":
    export(json.loads(sys.argv[1]), *(Path(p) for p in sys.argv[2:6]), sys.argv[6])
