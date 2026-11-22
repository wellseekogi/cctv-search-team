"""MEVA paired temporal-proposal and text-to-interval retrieval evaluation."""
import argparse
import json
from collections import Counter
from pathlib import Path

from embedding import create_encoder
from embedding.contracts import read_jsonl, write_json, write_jsonl
from team_search.__main__ import ROOT, get_config
from team_search.frame_state import gallery, rank_windows


def tiou(a, b):
    overlap = max(0, min(a["end_ms"], b["end_ms"]) - max(a["start_ms"], b["start_ms"]))
    return overlap / (a["end_ms"] - a["start_ms"] + b["end_ms"] - b["start_ms"] - overlap)


def rank_result(ranked, truth):
    first_center = first_interval = None
    for i, row in enumerate(ranked, 1):
        same = [t for t in truth if t["video_id"] == row["video_id"]]
        if first_center is None and any(t["start_ms"] <= row["timestamp_ms"] < t["end_ms"] for t in same):
            first_center = i
        if first_interval is None and any(tiou(row, t) >= 0.5 for t in same):
            first_interval = i
        if first_center is not None and first_interval is not None:
            break
    return {"rank": first_center, "interval_rank_50": first_interval}


def proposal_metrics(windows, activities):
    scores = [max((tiou(w, t) for w in windows if w["video_id"] == t["video_id"]), default=0)
              for t in activities]
    return {"annotated_activities": len(scores), "overlap": sum(s > 0 for s in scores),
            **{f"tiou_{round(thresh * 100)}": sum(s >= thresh for s in scores) for thresh in (0.3, 0.5, 0.7)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "runs/source-data")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/frame-state/cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/frame-state")
    parser.add_argument("--split", choices=("eval", "dev"), default="eval")
    parser.add_argument("--policy", choices=("state", "uniform"), default="state")
    parser.add_argument("--windows-per-minute", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--embedding-config", type=Path, default=ROOT / "configs/embedding.siglip2.yaml")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(min(4, torch.get_num_threads()))
    config = get_config(args)
    encoder = create_encoder(config)
    batch = config["batch_size_gpu" if encoder.device.startswith("cuda") else "batch_size_cpu"]
    windows, vectors, folders = gallery(ROOT / "data", args.data_root, args.cache, ["meva"], args.split,
                                        args.windows_per_minute, encoder, batch, args.policy == "uniform")
    dest = args.output_dir
    dest.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}-{args.policy}"
    write_jsonl(dest / (stem + "-windows.jsonl"), windows)
    data = ROOT / "data/meva"
    # Read human judgments only after the gallery is finalized.
    qs = [q for q in read_jsonl(data / "queries.jsonl") if q["split"] == args.split and q["modality"] == "text"]
    labels = read_jsonl(data / "qrels.jsonl")
    activities = [t for t in read_jsonl(data / "targets.jsonl")
                  if t["target_type"] == "activity" and t["evaluable"] and t["split"] == args.split]
    ko = json.loads((ROOT / "evaluation/meva_ko.json").read_text(encoding="utf8"))
    results = []
    for q in qs:
        qid = q["query_id"]
        truth = [t for t in labels if t["query_id"] == qid and t["relevance"] > 0]
        if not truth:
            raise ValueError(f"No positive labels for {qid}")
        for lang, vec in zip(("ko", "en"), encoder.encode_text([ko[qid], q["text"]]), strict=True):
            for method in ("three_frames", "center_frame"):
                ranked = rank_windows(windows, vectors, vec, center_only=method == "center_frame")
                results.append(dict(query_id=qid, language=lang, kind=q["kind"], method=method,
                                    **rank_result(ranked, truth),
                                    top5=[[r["video_id"], r["timestamp_ms"]] for r in ranked[:5]]))
    report = dict(policy=args.policy, split=args.split, rate=args.windows_per_minute, model=encoder.metadata,
                  windows=len(windows), sources=Counter(w["source"] for w in windows),
                  costs=[json.loads((f / "cost.json").read_text(encoding="utf8")) for f in folders],
                  proposals=proposal_metrics(windows, activities), rows=results)
    write_json(dest / (stem + ".json"), report)
    print(stem, report["windows"], report["proposals"], flush=True)


if __name__ == "__main__":
    main()
