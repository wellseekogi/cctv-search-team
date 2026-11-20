"""Summarize archived HTTP trials without dropping failures or counting repetitions as new questions."""

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def time_verdict(text, expected_ms):
    mentions = re.findall(r"(\d+(?:\.\d+)?)\s*초", text)
    if not mentions:
        return "unspecified"
    return "matched" if all(abs(float(t) * 1000 - expected_ms) <= 50 for t in mentions) else "wrong"


def compare(trials):
    groups = defaultdict(list)
    for trial in trials:
        groups[trial["query"]].append(trial)
    report = {}
    for key, group in sorted(groups.items()):
        good = [r for r in group if r["status"] == "generated" and r["results"]]
        ranks = {tuple(hit["item_id"] for hit in r["results"]) for r in good}
        scores = {tuple(hit["score"] for hit in r["results"]) for r in good}
        times = defaultdict(int)
        for r in good:
            expected = r.get("previous_second", r["results"][0])
            times[time_verdict(r["answer"], expected["playback_time_ms"])] += 1
        group_report = {
            "requests": len(group),
            "generated": len(good),
            "distinct_rankings": len(ranks),
            "distinct_score_lists": len(scores),
            "distinct_answers": len({r["answer"] for r in good}),
            "answer_time": dict(times),
        }
        if good:
            group_report["first_result"] = good[0]["results"][0]
        if good and "previous_second" in good[0]:
            group_report["previous_second_still_second"] = sum(
                len(r["results"]) >= 2 and r["results"][1]["item_id"] == r["previous_second"]["item_id"]
                for r in good
            )
        report[key] = group_report
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=Path, required=True)
    p.add_argument("--follow-ups", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    trials, follow = rows(args.trials), rows(args.follow_ups)
    latency = sorted(r["seconds"] for r in trials)
    versions = sorted({r["version"] for r in trials + follow if "version" in r})
    if len(versions) != 1:
        raise ValueError("Repeatability requires a fixed gallery")
    qrels = rows(ROOT / "data/meva/qrels.jsonl")
    report = {
        "index_version": versions[0],
        "candidates": sorted({r["candidates"] for r in trials if "candidates" in r}),
        "requests": len(trials),
        "follow_up_requests": len(follow),
        "median_seconds": statistics.median(latency),
        "p95_seconds": latency[math.ceil(0.95 * len(latency)) - 1],
        "single": compare(trials),
        "follow_up": compare(follow),
    }
    for key, group in report["single"].items():
        if not key.startswith("mq_"):
            continue
        first = next((r for r in trials if r["query"] == key and r.get("results")), None)
        truths = [t for t in qrels if t["query_id"] == key and t["relevance"] > 0]
        group["first_trial_temporal_rank_in_top5"] = (
            next(
                (
                    i
                    for i, h in enumerate(first["results"], 1)
                    if any(
                        h["video_id"].removeprefix("meva_") == t["video_id"]
                        and t["start_ms"] <= h["timestamp_ms"] < t["end_ms"]
                        for t in truths
                    )
                ),
                None,
            )
            if first
            else None
        )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
