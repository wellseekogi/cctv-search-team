"""Exercise the real web search and Qwen endpoint with interleaved, independent requests."""

import argparse
import json
import random
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def request_search(url, prompt, history):
    payload = {"text": prompt, "top_k": 5, "representation": "combined", "history": history}
    req = Request(
        url.rstrip("/") + "/search",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urlopen(req, timeout=120) as response:
            reply = json.load(response)
        keys = ("item_id", "video_id", "frame_index", "timestamp_ms", "playback_time_ms", "score")
        row = {
            "status": reply["llm_status"],
            "answer": reply["answer"],
            "version": reply["index_version"],
            "candidates": reply["total_candidates"],
            "results": [{k: r[k] for k in keys} for r in reply["results"]],
        }
    except (URLError, TimeoutError) as exc:
        row = {"status": "error", "error": str(exc)}
    return {**row, "seconds": round(time.perf_counter() - start, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8780")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--follow-up-from", type=Path, help="Use first-round web exchanges as conversation history"
    )
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error("At least two rounds are needed")
    translated = json.loads((ROOT / "evaluation/meva_ko.json").read_text(encoding="utf-8"))
    with (ROOT / "data/meva/queries.jsonl").open(encoding="utf-8") as f:
        dev = [q for q in map(json.loads, f) if q["split"] == "dev" and q["modality"] == "text"]
    prompts = {q["query_id"]: translated[q["query_id"]] for q in dev}
    prompts.update(
        cafe="카페에서 테이블 옆에 있는 사람을 찾아줘.",
        cafe_reworded="탁자 곁에 사람이 있는 카페 장면을 보여 줘.",
        absent_ski="눈 덮인 스키장에서 스키를 타는 사람을 찾아줘.",
        absent_pool="수영장 안에서 헤엄치는 사람을 찾아줘.",
        absent_space="달 표면에서 우주복을 입고 걷는 사람을 찾아줘.",
    )
    parents = {}
    if args.follow_up_from:
        with args.follow_up_from.open(encoding="utf-8") as inputs:
            parents = {
                row["query"]: row
                for row in map(json.loads, inputs)
                if row["round"] == 1 and row["status"] == "generated"
            }
        prompts = {
            key: "방금 추천한 결과 중 두 번째 후보는 몇 초부터 보면 돼?"
            for key in ("cafe", "mq_d02", "mq_d04", "mq_d07", "mq_d14")
        }
        if not prompts.keys() <= parents.keys():
            raise ValueError("Missing first-round responses for follow-up test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        for n in range(args.rounds):
            order = list(prompts)
            random.Random(913 + n).shuffle(order)
            for key in order:
                row = {"round": n + 1, "query": key, "text": prompts[key]}
                history = []
                if parents:
                    parent = parents[key]
                    history = [
                        {"role": "user", "content": parent["text"]},
                        {"role": "assistant", "content": parent["answer"]},
                    ]
                    row["previous_second"] = parent["results"][1]
                row.update(request_search(args.base_url, prompts[key], history))
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
            print(f"Round {n + 1}/{args.rounds} recorded", flush=True)


if __name__ == "__main__":
    main()
