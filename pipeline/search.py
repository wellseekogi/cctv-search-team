"""질의 인코딩 → 순위 → predictions.jsonl (F 평가기 형식)."""
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict

from .core import ROOT, percentile, read_jsonl


def eval_queries(dataset, split, video_ids):
    """이 run의 영상에 정답이 있는 텍스트 질의."""
    base = ROOT / "data" / dataset
    if not ((base / "qrels.jsonl").is_file() and (base / "queries.jsonl").is_file()):
        return []
    wanted = {r["query_id"] for r in read_jsonl(base / "qrels.jsonl") if r.get("video_id") in video_ids}
    return [q for q in read_jsonl(base / "queries.jsonl")
            if q.get("modality") == "text" and q.get("split") == split and q.get("query_id") in wanted]


def _track_index(tracks):
    by = defaultdict(list)
    for t in tracks:
        by[t["video_id"]].append((t["timestamp_ms"], t["track_uid"]))
    return {v: ([r[0] for r in sorted(rows)], [r[1] for r in sorted(rows)]) for v, rows in by.items()}


def _tracks_near(index, video_id, stamp, tolerance):
    times, uids = index.get(video_id, ([], []))
    low, high = bisect_left(times, stamp - tolerance), bisect_right(times, stamp + tolerance)
    return sorted(set(uids[low:high]))


def predict(engine, encoder, queries, tracks, representation, cap=5000, tolerance=100):
    """질의마다 트랙 순위를 만든다. 평가기는 질의당 1행의 results를 읽는다.
    object 행은 자기 트랙, scene 행은 같은 시각 ±tolerance ms에 관측된 트랙들로 풀어 첫 등장 순으로 합친다."""
    by_item = {r["item_id"]: r for r in engine.items}
    candidates = [r for r in engine.items if representation == "combined" or r["representation"] == representation]
    near = _track_index(tracks)
    encoder.encode_text(["warm up"])
    rows, latencies = [], []
    for q in queries:
        started = time.perf_counter()
        vector = encoder.encode_text([q["text"]])
        hits = engine.ranked_frames(vector, candidates, min(len(candidates), cap), text_search=True) if candidates else []
        seen, ranking = set(), []
        for hit in hits:
            row = by_item[hit["item_id"]]
            uids = ([row["track_uid"]] if row["representation"] == "object"
                    else _tracks_near(near, hit["video_id"], hit["timestamp_ms"], tolerance))
            for uid in uids:
                if uid is not None and uid not in seen:
                    seen.add(uid)
                    ranking.append((uid, hit["ranking_score"]))
        latencies.append((time.perf_counter() - started) * 1000)
        rows.append({"query_id": q["query_id"], "representation": representation,
                     "results": [{"rank": i, "track_uid": u, "score": round(float(s), 6)}
                                 for i, (u, s) in enumerate(ranking, 1)]})
    stats = {"n_queries": len(latencies), "warmup": True, "meets_20_query_rule": len(latencies) >= 20,
             "p50_ms": percentile(latencies, 0.5), "p95_ms": percentile(latencies, 0.95),
             "candidates": len(candidates)}
    return rows, stats