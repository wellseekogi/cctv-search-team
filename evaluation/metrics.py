"""검색 지표 (F 5 "지표 계산").

- Recall@K = 상위 K의 정답 트랙 수 / 해당 질의의 전체 정답 트랙 수
- MRR = 첫 정답 순위의 역수 평균 (전체 순위에 정답이 없으면 0)
- pred 순위: 같은 GT에 연결된 pred 조각은 첫 등장으로 합치고 순위를 다시 매긴다.
  미매칭 pred는 오답 후보로 유지한다.
- 분모: 전체 유효 질의 (predictions에 없는 질의도 0으로 포함)
"""
from __future__ import annotations

from collections import defaultdict


def collapse_ranking(track_uids, pred_to_gt=None):
    """순위 목록 → GT 기준 순위 (중복은 첫 등장만 유지).

    pred_to_gt=None 이면 GT 트랙 실험(입력이 이미 GT uid).
    """
    out, seen = [], set()
    for u in track_uids:
        if u is None:
            continue
        key = u
        if pred_to_gt is not None:
            g = pred_to_gt.get(u)
            key = g if g is not None else f"unmatched::{u}"
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def query_metrics(ranking, relevant, ks=(5, 10)):
    rel = set(relevant)
    first = None
    for i, u in enumerate(ranking, 1):
        if u in rel:
            first = i
            break
    m = {f"recall@{k}": (len(rel & set(ranking[:k])) / len(rel)) if rel else None for k in ks}
    m["rr"] = 1.0 / first if first else 0.0
    m["first_relevant_rank"] = first
    m["n_relevant"] = len(rel)
    m["n_ranked"] = len(ranking)
    return m


def evaluate_rankings(queries, qrels, predictions, pred_to_gt=None, ks=(5, 10)):
    """queries: 평가 대상 질의 행, qrels: 정답 행, predictions: {query_id: [track_uid...]}.

    반환: (summary, per_query)
    """
    rel = defaultdict(list)
    for r in qrels:
        rel[r["query_id"]].append(r["gt_track_uid"])
    per_query = []
    for q in queries:
        qid = q["query_id"]
        if not rel.get(qid):
            continue                       # 정답 없는 질의는 평가 제외 (no_match는 UI 검사용)
        raw = predictions.get(qid)
        ranking = collapse_ranking(raw or [], pred_to_gt)
        m = query_metrics(ranking, rel[qid], ks)
        m.update({"query_id": qid, "modality": q.get("modality"), "split": q.get("split"),
                  "has_prediction": raw is not None})
        per_query.append(m)

    def agg(rows):
        if not rows:
            return {f"recall@{k}": None for k in ks} | {"mrr": None, "n_queries": 0}
        out = {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / len(rows) for k in ks}
        out["mrr"] = sum(r["rr"] for r in rows) / len(rows)
        out["n_queries"] = len(rows)
        out["n_missing_prediction"] = sum(1 for r in rows if not r["has_prediction"])
        return out

    summary = {"all": agg(per_query)}
    for mod in sorted({r["modality"] for r in per_query if r["modality"]}):
        summary[mod] = agg([r for r in per_query if r["modality"] == mod])
    return summary, per_query
