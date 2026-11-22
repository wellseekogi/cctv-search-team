"""Match ordered state windows to English text with the pretrained 8-frame X-CLIP."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import av
import numpy as np
import torch
from transformers import XCLIPModel, XCLIPProcessor

from embedding.contracts import file_sha256, read_jsonl, write_json


WEIGHTS_SHA256 = "abf286e8cdd0612761c3e42d3a55eca998382dfa67a04a0f3fdcdfa4f150cdbb"
ORDERS = ("ordered", "reversed", "repeated")


class StateVerifier:
    """Score all query texts for one clip in a single video forward pass."""

    def __init__(self, model_dir, device):
        if file_sha256(Path(model_dir) / "model.safetensors") != WEIGHTS_SHA256:
            raise ValueError("expected microsoft/xclip-base-patch32 revision a2e27a78")
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = XCLIPModel.from_pretrained(model_dir, local_files_only=True,
                                                torch_dtype=self.dtype).to(self.device).eval()
        self.processor = XCLIPProcessor.from_pretrained(model_dir, local_files_only=True)

    @torch.inference_mode()
    def score(self, frames, prompts):
        if len(frames) != self.model.config.vision_config.num_frames:
            raise ValueError("X-CLIP requires exactly eight RGB frames")
        text = self.processor.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        pixels = self.processor(videos=[frames], return_tensors="pt").pixel_values
        logits = self.model(**text, pixel_values=pixels.to(self.device, self.dtype)).logits_per_video
        return logits[0].float().cpu().numpy()


def clip_indices(window):
    first, center, last = window["frame_indices"]
    if not 0 <= first <= center <= last:
        raise ValueError("invalid state window frame indices")
    return np.linspace(first, last, 8).round().astype(int).tolist()


def read_frames(path, needed):
    """Decode once per camera and letterbox full images to preserve the scene edges."""
    frames = {}
    last = max(needed)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 3
        for index, frame in enumerate(container.decode(stream)):
            if index in needed:
                scale = 224 / max(frame.width, frame.height)
                h, w = max(1, round(frame.height * scale)), max(1, round(frame.width * scale))
                rgb = frame.reformat(width=w, height=h, format="rgb24").to_ndarray()
                square = np.full((224, 224, 3), 127, dtype=np.uint8)
                top, left = (224 - h) // 2, (224 - w) // 2
                square[top:top + h, left:left + w] = rgb
                frames[index] = square
            if index == last:
                break
    if needed != frames.keys():
        raise ValueError(f"Missing {len(needed - frames.keys())} frames in {path}")
    return frames


def score_windows(windows, media, verifier, prompts, controls=False):
    grouped = defaultdict(list)
    for row, window in enumerate(windows):
        grouped[window["video_id"]].append(row)
    orders = ORDERS if controls else ("ordered",)
    scores = {order: np.empty((len(windows), len(prompts))) for order in orders}
    inference = 0.0
    for vid, rows in grouped.items():
        needed = {i for row in rows for i in clip_indices(windows[row])}
        if controls:
            needed.update(windows[row]["frame_indices"][1] for row in rows)
        frames = read_frames(media[vid], needed)
        for row in rows:
            clip = [frames[i] for i in clip_indices(windows[row])]
            evidence = {"ordered": clip}
            if controls:
                center = frames[windows[row]["frame_indices"][1]]
                evidence.update(reversed=clip[::-1], repeated=[center] * len(clip))
            start = perf_counter()
            for order, video in evidence.items():
                scores[order][row] = verifier.score(video, prompts)
            inference += perf_counter() - start
        print(f"Scored {len(rows)} windows on {vid}", flush=True)
    return scores, inference


def search_video(args):
    from integrations.f_metadata import image_exclusions
    from team_search.baseline import source_video
    from team_search.frame_state import interval_rows

    if args.windows_per_minute <= 0:
        raise ValueError("windows-per-minute must be positive")
    datasets = ("meva", "virat") if args.dataset == "all" else (args.dataset,)
    media, windows = {}, []
    for dataset in datasets:
        metadata = args.metadata_root / dataset
        queries = read_jsonl(metadata / "queries.jsonl")
        for m in read_jsonl(metadata / "manifest.jsonl"):
            if args.split != "all" and args.split != m["split"]:
                continue
            path = source_video(args.metadata_root, args.data_root, m)
            vid = m["video_id"]
            media[vid] = path
            exclusions = image_exclusions(queries, dict(m, f_video_id=vid))
            duration = m["available_end_ms"] - m["available_start_ms"]
            budget = max(1, round(duration * args.windows_per_minute / 60000))
            windows.extend(interval_rows(path, m, exclusions, budget, uniform=False))
    if not windows:
        print("검색 가능한 구간이 없습니다.")
        return
    verifier = StateVerifier(args.video_model, args.device)
    scores, _ = score_windows(windows, media, verifier, [args.question])
    for rank, index in enumerate(np.argsort(-scores["ordered"][:, 0], kind="stable")[:args.top_k], 1):
        row = windows[index]
        print(f"{rank}. {row['video_id']} | {row['start_ms']/1000:.2f}–{row['end_ms']/1000:.2f}초"
              f" | X-CLIP 로짓 {scores['ordered'][index, 0]:.4f} | {row['source']}")


def run(args):
    windows = read_jsonl(args.windows)
    manifests = {m["video_id"]: m for m in read_jsonl(args.manifest)}
    vids = {w["video_id"] for w in windows}
    splits = {manifests[vid]["split"] for vid in vids}
    if len(splits) != 1:
        raise ValueError("supply a nonempty gallery from a single split")
    queries = [q for q in read_jsonl(args.queries) if q["split"] in splits
               and q["modality"] == "text" and q["language"] == "en"]
    if not queries:
        raise ValueError("no English text queries for this split")
    media = {vid: args.media / manifests[vid]["relative_path"] for vid in vids}
    for vid, path in media.items():
        if file_sha256(path) != manifests[vid]["file_sha256"]:
            raise ValueError(f"wrong media checksum for {vid}")
    torch.set_num_threads(3)
    verifier = StateVerifier(args.model, args.device)
    prompts = [q["text"] for q in queries]
    if args.action_targets:
        from evaluation.state_action_pairs import ACTION_PAIRS, evaluate_pairs
        prompts.extend(p for pair in ACTION_PAIRS for _, p in pair)
    started = perf_counter()
    scores, inference = score_windows(windows, media, verifier, prompts, controls=True)
    output = dict(model="microsoft/xclip-base-patch32", revision="a2e27a78a2b5d802e894b8a1ef14f3a8ce490963",
                  weights_sha256=WEIGHTS_SHA256, windows_sha256=file_sha256(args.windows),
                  device=str(verifier.device), dtype=str(verifier.dtype), elapsed_sec=perf_counter() - started,
                  score_sec=inference, forward_passes=len(windows) * len(ORDERS), query_ids=[q["query_id"] for q in queries],
                  windows=windows, scores={order: values[:, :len(queries)].tolist() for order, values in scores.items()})
    if args.action_targets:
        logits = {order: values[:, len(queries):] for order, values in scores.items()}
        output["action_pairs"] = evaluate_pairs(windows, read_jsonl(args.action_targets), logits)
        output["action_pair_logits"] = {order: values.tolist() for order, values in logits.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, output)
    print(json.dumps({k: output[k] for k in ("device", "forward_passes", "score_sec", "elapsed_sec")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("windows", "manifest", "media", "queries", "model", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--action-targets", type=Path,
                        help="Optionally score four predefined action pairs, then compare with MEVA targets")
    run(parser.parse_args())
