"""보존·누락 분류 (F 6).

분모: targets.jsonl의 전체 대상 (evaluable=false 는 N/A로 따로 집계).
유효 관측 (valid observation):
  object: 대상 구간 안의 object 샘플 중 GT 박스와 IoU ≥ 0.5 이고,
          모델 입력(여백 crop → 짧은 변 224 resize → center crop)에서 보이는 GT 폭·높이가 각각 ≥ 8px.
  scene : 대상 구간 안의 scene 샘플 중 모델 전처리 후 남는 영역(center crop)이 GT 면적의 50% 이상을
          포함하고, 입력 좌표의 보이는 GT 폭·높이가 각각 ≥ 8px.
트랙 대상: 유효 관측 1개 이상이면 indexed.
활동 대상: 구간을 앞·중간·뒤 3등분, 각 부분에서 필수 참여자 모두의 유효 관측이 있고
           서로 다른 시점이 총 3개 이상이면 indexed.
누락 원인 (먼저 해당하는 것 하나):
  detection_missing  pred 실험에서 대상 GT에 연결된 pred 트랙이 없음
  sampling_missing   유효 관측이 되는 선택 샘플이 없음
  encoding_missing   유효 관측 샘플은 있으나 C의 items(벡터)에 없음
retrieval_missing 은 질의 단위: indexed 정답이 상위 K에 없을 때 (evaluate.py에서 계산).
"""
from __future__ import annotations

import collections
import math
from collections import defaultdict

from common import TimeIndex, intersect, iou


def _center_crop_region(w, h, size):
    """짧은 변을 size로 resize 후 center crop 했을 때 원본 좌표에서 남는 영역과 배율."""
    s = size / min(w, h)
    cw, ch = size / s, size / s
    x0, y0 = (w - cw) / 2.0, (h - ch) / 2.0
    return [x0, y0, x0 + cw, y0 + ch], s


def object_visible(sample_bbox, gt_bbox, frame_w, frame_h, cfg):
    """object 샘플이 GT를 유효하게 담는지 (IoU·입력 크기)."""
    if iou(sample_bbox, gt_bbox) < cfg["iou_threshold"]:
        return False, "iou"
    m = cfg.get("crop_margin_ratio", 0.10)
    x1, y1, x2, y2 = sample_bbox
    mw, mh = (x2 - x1) * m, (y2 - y1) * m
    crop = [max(0, math.floor(x1 - mw)), max(0, math.floor(y1 - mh)),
            min(frame_w, math.ceil(x2 + mw)), min(frame_h, math.ceil(y2 + mh))]
    cw, ch = crop[2] - crop[0], crop[3] - crop[1]
    reg, s = _center_crop_region(cw, ch, cfg["model_input_size"])
    reg = [reg[0] + crop[0], reg[1] + crop[1], reg[2] + crop[0], reg[3] + crop[1]]
    vis = intersect(gt_bbox, reg)
    if vis is None:
        return False, "not_visible"
    if (vis[2] - vis[0]) * s < cfg["min_input_side_px"] or (vis[3] - vis[1]) * s < cfg["min_input_side_px"]:
        return False, "too_small"
    return True, None


def scene_visible(gt_bbox, frame_w, frame_h, cfg):
    reg, s = _center_crop_region(frame_w, frame_h, cfg["model_input_size"])
    vis = intersect(gt_bbox, reg)
    if vis is None:
        return False, "cropped_out"
    area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    frac = (vis[2] - vis[0]) * (vis[3] - vis[1]) / area if area > 0 else 0
    if frac < cfg["min_visible_fraction"]:
        return False, "cropped_out"
    if (vis[2] - vis[0]) * s < cfg["min_input_side_px"] or (vis[3] - vis[1]) * s < cfg["min_input_side_px"]:
        return False, "too_small"
    return True, None


def valid_observations(target_uid, video_id, start_ms, end_ms, samples_by_video, gt_index,
                       manifest, cfg, pred_to_gt=None):
    """대상 GT 트랙의 유효 관측 샘플 목록 [(sample, representation)]."""
    m = manifest[video_id]
    W, H = m["width"], m["height"]
    tol = cfg.get("time_tolerance_ms", 100)
    out = []
    for s in samples_by_video.get(video_id, []):
        t = s["timestamp_ms"]
        if not (start_ms <= t < end_ms):
            continue
        g = gt_index.nearest(target_uid, t, tol)
        if g is None:
            continue
        rep = s["representation"]
        if rep == "object":
            if pred_to_gt is not None and s.get("track_uid") is not None:
                # pred 실험: 샘플 트랙이 다른 GT에 연결돼 있으면 IoU가 높아도 해당 대상의 관측이 아님
                linked = pred_to_gt.get(s["track_uid"])
                if linked is not None and linked != target_uid:
                    continue
            ok, _ = object_visible(s["bbox"], g["bbox"], W, H, cfg)
        elif rep == "scene":
            ok, _ = scene_visible(g["bbox"], W, H, cfg)
        else:
            ok = False
        if ok:
            out.append((s, rep))
    return out


def classify_targets(targets, samples, gt_rows, manifest, cfg, encoded_item_ids=None,
                     pred_to_gt=None, representations=("object", "scene")):
    """대상별 보존 판정. encoded_item_ids=None 이면 인코딩 검사를 생략(모두 인코딩된 것으로 간주).

    representations: 이 실험 인덱스에 포함된 표현 (E05=scene, E06=object, 나머지는 둘 다).
    """
    gt_index = TimeIndex(gt_rows, "track_uid")
    sv = defaultdict(list)
    for s in samples:
        if s["representation"] in representations:
            sv[s["video_id"]].append(s)
    detected_gt = set(pred_to_gt.values()) if pred_to_gt is not None else None

    results = []
    for t in targets:
        r = {"target_id": t["target_id"], "target_type": t["target_type"], "video_id": t["video_id"],
             "split": t["split"]}
        if not t.get("evaluable", True):
            r.update(status="N/A", reason=t.get("na_reason") or "not_evaluable")
            results.append(r)
            continue
        uids = t["gt_track_uids"]
        if t["target_type"] == "activity":
            uids = [p["track_uid"] for p in t.get("participants", []) if p.get("required", True)]
        if detected_gt is not None and not all(u in detected_gt for u in uids):
            r.update(status="missing", reason="detection_missing")
            results.append(r)
            continue

        if t["target_type"] == "track":
            obs = valid_observations(uids[0], t["video_id"], t["start_ms"], t["end_ms"], sv, gt_index,
                                     manifest, cfg, pred_to_gt)
            if not obs:
                r.update(status="missing", reason="sampling_missing")
            elif encoded_item_ids is not None and not any(s["item_id"] in encoded_item_ids for s, _ in obs):
                r.update(status="missing", reason="encoding_missing")
            else:
                enc = [o for o in obs if encoded_item_ids is None or o[0]["item_id"] in encoded_item_ids]
                r.update(status="indexed", reason=None, n_valid_obs=len(enc),
                         by_representation={rep: sum(1 for _, x in enc if x == rep) for rep in representations})
            results.append(r)
            continue

        # activity: 3등분 × 필수 참여자
        st, en = t["start_ms"], t["end_ms"]
        cuts = [st, st + (en - st) // 3, st + 2 * (en - st) // 3, en]
        required = [p["track_uid"] for p in t.get("participants", []) if p.get("required", True)]
        parts_ok, parts_ok_sampled, times = [], [], set()
        for i in range(3):
            ok_all, ok_all_sampled = True, True
            for u in required:
                obs = valid_observations(u, t["video_id"], cuts[i], cuts[i + 1], sv, gt_index,
                                         manifest, cfg, pred_to_gt)
                if not obs:
                    ok_all = ok_all_sampled = False
                    continue
                enc = [o for o in obs if encoded_item_ids is None or o[0]["item_id"] in encoded_item_ids]
                if not enc:
                    ok_all = False
                times.update(s["timestamp_ms"] for s, _ in enc)
            parts_ok.append(ok_all)
            parts_ok_sampled.append(ok_all_sampled)
        if all(parts_ok) and len(times) >= 3:
            r.update(status="indexed", reason=None, n_time_points=len(times))
        elif not all(parts_ok_sampled):
            r.update(status="missing", reason="sampling_missing", parts=parts_ok_sampled)
        elif len(times) < 3 and all(parts_ok):
            r.update(status="missing", reason="sampling_missing", n_time_points=len(times))
        else:
            r.update(status="missing", reason="encoding_missing", parts=parts_ok)
        results.append(r)
    return results


def summarize(results):
    out = {}
    for tt in ("track", "activity"):
        rows = [r for r in results if r["target_type"] == tt]
        ev = [r for r in rows if r["status"] != "N/A"]
        idx = sum(1 for r in ev if r["status"] == "indexed")
        reasons = defaultdict(int)
        for r in ev:
            if r["status"] == "missing":
                reasons[r["reason"]] += 1
        out[tt] = {
            "total_targets": len(ev), "indexed": idx,
            "missing_rate": (len(ev) - idx) / len(ev) if ev else None,
            "retention_rate": idx / len(ev) if ev else None,
            "missing_by_reason": dict(reasons),
            "n_not_applicable": len(rows) - len(ev),
            "na_reasons": dict(collections.Counter(r["reason"] for r in rows if r["status"] == "N/A")),
        }
    return out
