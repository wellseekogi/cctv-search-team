"""pred 트랙 ↔ GT 트랙 연결 (F 5 "Pred와 GT 연결").

1. 같은 영상에서 pred 관측 시각과 100ms 이내의 GT 프레임(가장 가까운 GT 시각)을 짝짓는다.
2. 그 프레임의 pred×GT 박스 IoU ≥ 0.5 쌍을 IoU 높은 순(동률: pred ID, GT ID 사전순)으로 일대일 배정.
3. pred 트랙마다 가장 많이 매칭된 GT 하나 선택 (동률: 평균 IoU 높은 순 → GT ID 사전순).
4. 매칭 프레임 ≥ 3 이고, GT 판정 가능 pred 관측 중 비율 ≥ 0.5 이면 연결.
   GT 판정 가능 관측 = 100ms 이내에 GT 프레임이 존재하는 pred 관측.
"""
from __future__ import annotations

import bisect
from collections import defaultdict

from common import iou


def _gt_frames(gt_rows):
    """video → sorted list of (timestamp_ms, [gt rows])."""
    by = defaultdict(lambda: defaultdict(list))
    for r in gt_rows:
        by[r["video_id"]][r["timestamp_ms"]].append(r)
    return {v: sorted(d.items()) for v, d in by.items()}


def _gt_annotated_ranges(gt_rows):
    """video → (min_ts, max_ts): GT 주석이 존재하는 시간 범위."""
    rng = {}
    for r in gt_rows:
        lo, hi = rng.get(r["video_id"], (r["timestamp_ms"], r["timestamp_ms"]))
        rng[r["video_id"]] = (min(lo, r["timestamp_ms"]), max(hi, r["timestamp_ms"]))
    return rng


def link_tracks(pred_rows, gt_rows, iou_threshold=0.5, tol_ms=100,
                min_match_frames=3, min_match_ratio=0.5, video_grid=None):
    """반환: track_links 행 목록 (pred 트랙마다 1행).

    video_grid: {video_id: [timestamp_ms, ...]} GT 판정 가능 시각 목록 (없으면 GT 주석 시각 사용).
    VIRAT처럼 '모든 객체 주석' 데이터는 GT 프레임에 객체가 하나도 없어도 판정 가능하므로
    convert 단계의 후보 격자 시각을 넘겨주는 것이 정확하다.
    """
    gtf = _gt_frames(gt_rows)
    gt_times = {v: [t for t, _ in lst] for v, lst in gtf.items()}
    gt_at = {v: dict(lst) for v, lst in gtf.items()}
    grid = video_grid or gt_times
    ann_range = _gt_annotated_ranges(gt_rows)

    pred_by_vid_time = defaultdict(lambda: defaultdict(list))
    n_pred = defaultdict(int)
    judgeable = defaultdict(int)
    for r in pred_rows:
        v, uid = r["video_id"], r["track_uid"]
        n_pred[(v, uid)] += 1
        ts_list = grid.get(v, [])
        i = bisect.bisect_left(ts_list, r["timestamp_ms"])
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts_list) and abs(ts_list[j] - r["timestamp_ms"]) <= tol_ms:
                if best is None or abs(ts_list[j] - r["timestamp_ms"]) < abs(best - r["timestamp_ms"]):
                    best = ts_list[j]
        if best is None:
            continue
        judgeable[(v, uid)] += 1
        pred_by_vid_time[v][best].append(r)

    # 프레임별 일대일 배정
    matches = defaultdict(list)            # (v, pred_uid) → [(gt_uid, iou)]
    for v, frames in pred_by_vid_time.items():
        for t, preds in frames.items():
            gts = gt_at.get(v, {}).get(t, [])
            pairs = []
            for p in preds:
                for g in gts:
                    s = iou(p["bbox"], g["bbox"])
                    if s >= iou_threshold:
                        pairs.append((-s, p["track_uid"], g["track_uid"], s))
            pairs.sort()
            used_p, used_g = set(), set()
            for _, pu, gu, s in pairs:
                if pu in used_p or gu in used_g:
                    continue
                used_p.add(pu)
                used_g.add(gu)
                matches[(v, pu)].append((gu, s))

    links = []
    for (v, pu), n in sorted(n_pred.items()):
        ms = matches.get((v, pu), [])
        cnt, sums = defaultdict(int), defaultdict(float)
        for gu, s in ms:
            cnt[gu] += 1
            sums[gu] += s
        best = None
        if cnt:
            best = sorted(cnt, key=lambda g: (-cnt[g], -sums[g] / cnt[g], g))[0]
        nm = cnt.get(best, 0) if best else 0
        nj = judgeable.get((v, pu), 0)
        ratio = nm / nj if nj else 0.0
        linked = best is not None and nm >= min_match_frames and ratio >= min_match_ratio
        links.append({
            "video_id": v, "pred_track_uid": pu,
            "gt_track_uid": best if linked else None,
            "best_gt_candidate": best, "n_pred_obs": n, "n_judgeable": nj,
            "n_matched": nm, "match_ratio": round(ratio, 4),
            "mean_iou": round(sums[best] / nm, 4) if nm else None,
            "linked": linked,
        })
    return links
