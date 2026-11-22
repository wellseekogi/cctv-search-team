"""Join the paired frozen MEVA runs and audit denominators before reporting differences."""
import argparse
import json
from collections import Counter
from pathlib import Path

from embedding.contracts import read_jsonl, write_json
from evaluation.benchmark_frame_state import proposal_metrics, rank_result
from team_search.__main__ import ROOT


def load(path):
    return json.loads(path.read_text(encoding="utf8"))


def coverage_ms(windows):
    total = 0
    for vid in {r["video_id"] for r in windows}:
        end = 0
        for row in sorted((r for r in windows if r["video_id"] == vid), key=lambda r: r["start_ms"]):
            total += max(0, row["end_ms"] - max(end, row["start_ms"]))
            end = max(end, row["end_ms"])
    return total


def compare(state_dir, baseline_path, video_path):
    state, uniform = (load(state_dir / f"eval-{method}.json") for method in ("state", "uniform"))
    baseline, video = load(baseline_path), load(video_path)
    if baseline["horizon_ms"] is not None:
        raise ValueError("The original B/A gallery must span the full eval videos")
    metadata = ROOT / "data/meva"
    queries = [q for q in read_jsonl(metadata / "queries.jsonl") if q["modality"] == "text" and q["split"] == "eval"]
    truths = read_jsonl(metadata / "qrels.jsonl")
    activities = [r for r in read_jsonl(metadata / "targets.jsonl")
                  if r["target_type"] == "activity" and r["evaluable"] and r["split"] == "eval"]
    windows = read_jsonl(state_dir / "eval-state-windows.jsonl")
    control = read_jsonl(state_dir / "eval-uniform-windows.jsonl")
    if Counter(r["video_id"] for r in windows) != Counter(r["video_id"] for r in control):
        raise ValueError("Unequal proposal budgets by camera")
    videos = [m for m in read_jsonl(metadata / "manifest.jsonl") if m["split"] == "eval"]
    sources = {m["video_id"]: m["file_sha256"] for m in videos}
    for m in videos:
        expected = len(range(m["available_start_ms"], m["available_end_ms"], 200))
        if baseline.get("b", {}).get(m["video_id"], {}).get("sampled_frames") != expected:
            raise ValueError(f"Incomplete B handoff for {m['video_id']}")
    for report in (state, uniform):
        if {r["sha256"] for r in report["costs"]} != set(sources.values()):
            raise ValueError("Window gallery sources differ from the eval split")
    if baseline["source_sha256"] != sources or video["windows"] != windows:
        raise ValueError("Original gallery or video-model evidence differs from paired evaluation")
    query_ids = [q["query_id"] for q in queries]
    if video["query_ids"] != query_ids:
        raise ValueError("Video prompts differ from the English benchmark")
    query_rows = []
    duration_ceiling = Counter()
    for q in queries:
        qid = q["query_id"]
        truth = [r for r in truths if r["query_id"] == qid and r["relevance"] > 0]
        # Even identical centers cannot yield IoU >= .5 outside this duration ratio.
        if any(1000 <= t["end_ms"] - t["start_ms"] <= 8000 for t in truth):
            duration_ceiling[q["kind"]] += 1
        for lang in ("ko", "en"):
            def match(report, method=None):
                found = [r for r in report["rows"] if r["query_id"] == qid and r["language"] == lang
                         and (method is None or r["method"] == method)]
                if len(found) != 1:
                    raise ValueError(f"Expected exactly one {qid}/{lang}/{method} result")
                return {k: found[0][k] for k in ("rank", "interval_rank_50") if k in found[0]}

            ranks = {"original": match(baseline), "uniform": match(uniform, "three_frames"),
                     "state": match(state, "three_frames"), "state_center_only": match(state, "center_frame")}
            original = next(r for r in baseline["rows"] if r["query_id"] == qid and r["language"] == lang)
            ranks["original_matched_40"] = {"rank": original["rank_matched_40"]}
            if lang == "en":
                j = query_ids.index(qid)
                for order, scores in video["scores"].items():
                    if len(scores) != len(windows):
                        raise ValueError("Incomplete video scores")
                    ordered = sorted(range(len(windows)), key=lambda i: -scores[i][j])
                    ranks["video_" + order] = rank_result([windows[i] for i in ordered], truth)
            query_rows.append(dict(query_id=qid, language=lang, kind=q["kind"], ranks=ranks))
    counts = []
    for lang in ("ko", "en"):
        for kind in ("all", "event", "appearance"):
            subset = [r for r in query_rows if r["language"] == lang and (kind == "all" or r["kind"] == kind)]
            for method in subset[0]["ranks"]:
                entries = [r["ranks"][method] for r in subset]
                counts.append(dict(language=lang, kind=kind, method=method, total=len(entries),
                                   **{f"hit{k}": sum(e["rank"] is not None and e["rank"] <= k for e in entries)
                                      for k in (1, 5, 10)}, available=sum(e["rank"] is not None for e in entries),
                                   interval_hit5=sum((e["interval_rank_50"] or float("inf")) <= 5 for e in entries)
                                   if "interval_rank_50" in entries[0] else None))
    return dict(base_commit="82b4e45", source_sha256=sources, queries=query_rows, counts=counts,
                original_items=baseline["items"], original_representations=baseline["representations"],
                original_by_camera=baseline["items_by_camera"],
                original_equal_positions={vid: {key: detail[key] for key in ("frames", "items")}
                                          for vid, detail in baseline["matched_40"].items()},
                b_frames_by_camera={vid: detail["sampled_frames"] for vid, detail in baseline["b"].items()},
                changed_top5_under_f_clock=[(r["query_id"], r["language"]) for r in baseline["rows"]
                                            if ((r["rank"] or float("inf")) <= 5)
                                            != ((r["rank_f_cfr"] or float("inf")) <= 5)],
                original_unique_frames=baseline["unique_frames"], windows=len(windows),
                duration_only_ceiling_50=dict(duration_ceiling),
                proposals={"state": state["proposals"], "uniform": uniform["proposals"]},
                union_coverage_ms={"state": coverage_ms(windows), "uniform": coverage_ms(control)},
                proposals_by_camera={vid: {method: proposal_metrics(w, [t for t in activities if t["video_id"] == vid])
                                           for method, w in (("state", windows), ("uniform", control))}
                                     for vid in sources},
                state_sources=state["sources"], state_costs=state["costs"], uniform_costs=uniform["costs"],
                action_pairs=video.get("action_pairs"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", type=Path, default=ROOT / "runs/frame-state")
    p.add_argument("--baseline", type=Path, default=ROOT / "runs/state-baseline/full-results.json")
    p.add_argument("--video", type=Path, default=ROOT / "runs/frame-state/xclip-scores-action-pairs.json")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = compare(args.state_dir, args.baseline, args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    for row in result["counts"]:
        if row["kind"] == "event":
            print(row["language"], row["method"], row["hit5"], "/", row["total"])
