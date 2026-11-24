"""Compare the ABCD baseline and pixel attention on the same MEVA eval queries."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from embedding import create_encoder
from embedding.contracts import file_sha256, read_jsonl, write_json
from evaluation.benchmark_frame_state import rank_result
from team_search.__main__ import ROOT, get_config
from team_search.frame_state import gallery, rank_windows


def compare(args):
    data = ROOT / "data/meva"
    sources = {v["video_id"]: v["file_sha256"] for v in read_jsonl(data / "manifest.jsonl") if v["split"] == "eval"}
    texts = {q["query_id"]: q for q in read_jsonl(data / "queries.jsonl")
             if q["split"] == "eval" and q["modality"] == "text"}
    labels = read_jsonl(data / "qrels.jsonl")
    korean = json.loads((ROOT / "evaluation/meva_ko.json").read_text(encoding="utf8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf8"))
    if baseline["horizon_ms"] is not None or baseline["source_sha256"] != sources:
        raise ValueError("ABCD results must cover the complete MEVA eval videos")
    pairs = [(r["query_id"], r["language"], r["kind"]) for r in baseline["rows"]]
    expected = [(qid, lang, q["kind"]) for qid, q in texts.items() for lang in ("ko", "en")]
    if Counter(pairs) != Counter(expected):
        raise ValueError("ABCD results must contain one rank for each MEVA eval query and language")

    config = get_config(args)
    encoder = create_encoder(config)
    batch = config["batch_size_gpu" if encoder.device.startswith("cuda") else "batch_size_cpu"]
    windows, vectors, _ = gallery(ROOT / "data", args.data_root, args.pixel_cache, ("meva",),
                                  "eval", 8, encoder, batch, pixel_attention=True)

    rows = []
    for row in baseline["rows"]:
        qid, lang = row["query_id"], row["language"]
        prompt = korean[qid] if lang == "ko" else texts[qid]["text"]
        truth = [t for t in labels if t["query_id"] == qid and t["relevance"] > 0]
        if not truth:
            raise ValueError(f"No positive labels for {qid}")
        ranked = rank_windows(windows, vectors, encoder.encode_text([prompt])[0], pixel_attention=True)
        rows.append(dict(query_id=qid, language=lang, kind=row["kind"],
                         abcd_rank=row["rank"], pixel_attention_rank=rank_result(ranked, truth)["rank"]))

    counts = []
    for lang in ("ko", "en"):
        for kind in ("all", "event", "appearance"):
            subset = [r for r in rows if r["language"] == lang and (kind == "all" or r["kind"] == kind)]
            for method in ("abcd", "pixel_attention"):
                ranks = [r[method + "_rank"] for r in subset if r[method + "_rank"] is not None]
                counts.append(dict(language=lang, kind=kind, method=method, total=len(subset),
                                   available=len(ranks), **{f"hit{k}": sum(r <= k for r in ranks)
                                                           for k in (1, 5, 10, 15)}))
    return dict(dataset="meva", split="eval", pixel_model=encoder.metadata, source_sha256=sources,
                abcd=dict(results_sha256=file_sha256(args.baseline), indexes=baseline["indexes"],
                          items=baseline["items"], positions=baseline["unique_frames"]),
                pixel_attention=dict(windows=len(windows), windows_by_video=Counter(w["video_id"] for w in windows),
                                     image_encodes=6 * len(windows)),
                counts=counts, rows=rows)


def main():
    shared_runs = ROOT.parent / "cctv-search-full/runs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=shared_runs / "source-data")
    parser.add_argument("--baseline", type=Path, default=shared_runs / "state-baseline/full-results.json",
                        help="Full results from evaluation.frame_state_baseline")
    parser.add_argument("--pixel-cache", type=Path, default=ROOT / "runs/frame-state/cache")
    parser.add_argument("--embedding-config", type=Path, default=ROOT / "configs/embedding.siglip2.yaml")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, help="Write results and paired ranks to JSON")
    args = parser.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    result = compare(args)
    print(f"ABCD: {result['abcd']['positions']} positions. Pixel attention: {result['pixel_attention']['windows']} windows.")
    print("Language  Queries     Method            Top 1  Top 5  Top 10  Top 15  Available")
    for row in result["counts"]:
        scores = "  ".join(f"{row[key]:2}/{row['total']:<2}" for key in ("hit1", "hit5", "hit10", "hit15", "available"))
        print(f"{row['language']:8}  {row['kind']:10}  {row['method']:16}  {scores}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)


if __name__ == "__main__":
    main()
