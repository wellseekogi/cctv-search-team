"""평가기 연결 검사용 모의 실행 폴더 생성 (F 단독 smoke, 실제 실험 결과 아님).

A·C·D 산출물이 나오기 전, 같은 파일 형식으로 evaluate.py 전체 경로를 검사한다.
  samples.jsonl     고정 간격 정책(기본 2000ms) 모의: 후보 시각마다 scene 1개 + GT object
                    (이미지 질의 제거 구간 적용)
  items.jsonl       samples 중 1%를 인코딩 실패로 제외
  predictions.jsonl 질의별 무작위 순위(seed 고정) + oracle 순위 두 가지
  tracks.jsonl      (--pred) GT 박스를 흔들고 조각낸 pseudo-pred 관측
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, read_jsonl, write_jsonl  # noqa: E402



def h(*parts):
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="virat")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--step-ms", type=int, default=2000)
    ap.add_argument("--ranking", choices=["random", "oracle"], default="random")
    ap.add_argument("--pred", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rnd = random.Random(42)
    out = ROOT / args.out

    DATA = ROOT / "data" / args.dataset
    man = {m["video_id"]: m for m in read_jsonl(DATA / "manifest.jsonl") if m["split"] == args.split}
    gt = [r for r in read_jsonl(DATA / "tracks_gt.jsonl")
          if r["video_id"] in man and r["class_name"] in ("person", "car", "vehicle", "bicycle")]
    excl = [e for e in read_jsonl(DATA / "image_query_exclusions.jsonl") if e["video_id"] in man]
    by_vt = defaultdict(list)
    times = defaultdict(set)
    for r in gt:
        by_vt[(r["video_id"], r["timestamp_ms"])].append(r)
        times[r["video_id"]].add(r["timestamp_ms"])

    # pseudo-pred (박스 흔들기 + 트랙 조각내기)
    pred = []
    gt2pred = defaultdict(list)
    if args.pred:
        for r in gt:
            if rnd.random() < 0.15:          # 15% 미검출
                continue
            x1, y1, x2, y2 = r["bbox"]
            w, hh = x2 - x1, y2 - y1
            j = lambda s: rnd.uniform(-0.08, 0.08) * s
            W, H = man[r["video_id"]]["width"], man[r["video_id"]]["height"]
            b = [max(0, x1 + j(w)), max(0, y1 + j(hh)), min(W, x2 + j(w)), min(H, y2 + j(hh))]
            frag = r["timestamp_ms"] // 20000          # 20초마다 ID가 바뀌는 조각
            lid = f"p{r['local_track_id'][1:]}_{frag}"
            gt2pred[r["track_uid"]].append(f"smoke01:{r['video_id']}:{lid}")
            pred.append({**r, "source": "pred", "track_source_id": "smoke01", "local_track_id": lid,
                         "track_uid": f"smoke01:{r['video_id']}:{lid}", "bbox": [round(v, 1) for v in b],
                         "confidence": 0.9})
        write_jsonl(out / "tracks.jsonl", pred)
    obs_src = pred if args.pred else gt
    by_vt_src = defaultdict(list)
    for r in obs_src:
        by_vt_src[(r["video_id"], r["timestamp_ms"])].append(r)

    samples = []
    for vid, m in man.items():
        ts_sorted = sorted(times[vid])
        t = 0
        while t < m["duration_ms"]:
            # 가장 가까운 후보 시각
            near = min(ts_sorted, key=lambda x: abs(x - t)) if ts_sorted else None
            t += args.step_ms
            if near is None or abs(near - (t - args.step_ms)) > 100:
                continue
            if any(e["video_id"] == vid and e["start_ms"] <= near < e["end_ms"] for e in excl):
                continue
            samples.append({"item_id": "scn_" + h(vid, near), "video_id": vid, "timestamp_ms": near,
                            "representation": "scene", "track_uid": None, "bbox": None,
                            "event_ids": [], "reasons": ["uniform"], "policy": f"uniform_{args.step_ms}"})
            for r in by_vt_src[(vid, near)]:
                samples.append({"item_id": "obj_" + h(vid, near, r["track_uid"]), "video_id": vid,
                                "timestamp_ms": near, "representation": "object",
                                "track_uid": r["track_uid"], "bbox": r["bbox"], "event_ids": [],
                                "reasons": ["uniform"], "policy": f"uniform_{args.step_ms}"})
    items = [{"item_id": s["item_id"]} for s in samples if rnd.random() > 0.01]

    queries = [q for q in read_jsonl(DATA / "queries.jsonl") if q["split"] == args.split]
    rel = defaultdict(list)
    for r in read_jsonl(DATA / "qrels.jsonl"):
        rel[r["query_id"]].append(r["gt_track_uid"])
    cand = sorted({s["track_uid"] for s in samples if s["track_uid"]})
    preds = []
    for q in queries:
        order = cand[:]
        rnd.shuffle(order)
        if args.ranking == "oracle":
            want = []
            for g in rel[q["query_id"]]:
                want += sorted(set(gt2pred[g])) if args.pred else [g]
            order = [u for u in want if u in cand] + [u for u in order if u not in want]
        preds.append({"query_id": q["query_id"], "representation": "combined",
                      "results": [{"rank": i + 1, "track_uid": u, "score": 1.0 / (i + 1)}
                                  for i, u in enumerate(order)]})
    write_jsonl(out / "samples.jsonl", samples)
    write_jsonl(out / "items.jsonl", items)
    write_jsonl(out / "predictions.jsonl", preds)
    print(f"{out}: samples={len(samples)} items={len(items)} queries={len(preds)} pred_obs={len(pred)}")


if __name__ == "__main__":
    main()
