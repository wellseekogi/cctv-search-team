"""평가 코드 손계산 대조 (F 5, F 7 완료 확인).

실행: python -m pytest evaluation/tests -q   (또는 python evaluation/tests/test_evaluation.py)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matching import link_tracks  # noqa: E402
from metrics import collapse_ranking, evaluate_rankings, query_metrics  # noqa: E402
from retention import classify_targets, object_visible, scene_visible, summarize  # noqa: E402

CFG = {"iou_threshold": 0.5, "min_match_frames": 3, "min_match_ratio": 0.5, "min_visible_fraction": 0.5,
       "min_input_side_px": 8, "time_tolerance_ms": 100, "model_input_size": 224, "crop_margin_ratio": 0.10}


# ---------------------------------------------------------------- 명세서 F 5 손계산
def test_spec_example_rank1_rank4_miss():
    """정답 1위·4위·미검색 세 질의: Recall@5=2/3, MRR=0.4167."""
    queries = [{"query_id": q, "modality": "text", "split": "dev"} for q in ("q1", "q2", "q3")]
    qrels = [{"query_id": "q1", "gt_track_uid": "A"}, {"query_id": "q2", "gt_track_uid": "B"},
             {"query_id": "q3", "gt_track_uid": "C"}]
    preds = {"q1": ["A", "x", "y", "z", "w"],
             "q2": ["x", "y", "z", "B", "w"],
             "q3": ["x", "y", "z", "w", "v"]}
    s, _ = evaluate_rankings(queries, qrels, preds)
    assert math.isclose(s["all"]["recall@5"], 2 / 3)
    assert round(s["all"]["mrr"], 4) == 0.4167


def test_multi_relevant_recall():
    m = query_metrics(["a", "G1", "b", "c", "d", "e", "G2"], ["G1", "G2", "G3"])
    assert math.isclose(m["recall@5"], 1 / 3)
    assert math.isclose(m["recall@10"], 2 / 3)
    assert m["rr"] == 0.5


def test_missing_prediction_counts_zero():
    queries = [{"query_id": "q1", "modality": "text"}, {"query_id": "q2", "modality": "image"}]
    qrels = [{"query_id": "q1", "gt_track_uid": "A"}, {"query_id": "q2", "gt_track_uid": "B"}]
    s, _ = evaluate_rankings(queries, qrels, {"q1": ["A"]})
    assert s["all"]["n_queries"] == 2 and s["all"]["mrr"] == 0.5
    assert s["all"]["n_missing_prediction"] == 1


def test_pred_fragments_merge_and_unmatched_kept():
    """같은 GT에 연결된 pred 조각은 첫 등장으로 합치고, 미매칭 pred는 오답으로 유지."""
    p2g = {"p1": "G1", "p2": "G1", "p3": "G2"}
    r = collapse_ranking(["p9", "p1", "p2", "p8", "p3"], p2g)
    assert r == ["unmatched::p9", "G1", "unmatched::p8", "G2"]
    m = query_metrics(r, ["G2"])
    assert m["first_relevant_rank"] == 4


# ---------------------------------------------------------------- pred-GT 연결
def _obs(uid, vid, t, box):
    return {"track_uid": uid, "video_id": vid, "timestamp_ms": t, "bbox": box}


def test_link_basic_and_threshold():
    gt = [_obs("G1", "v", t, [100, 100, 200, 200]) for t in range(0, 1000, 200)]
    gt += [_obs("G2", "v", t, [400, 100, 500, 200]) for t in range(0, 1000, 200)]
    pred = [_obs("P1", "v", t + 30, [105, 100, 205, 200]) for t in range(0, 1000, 200)]   # 5 매칭
    pred += [_obs("P2", "v", t, [400, 100, 500, 200]) for t in (0, 200)]                  # 2 매칭 → 미연결
    pred += [_obs("P3", "v", t, [700, 700, 800, 800]) for t in range(0, 1000, 200)]       # 0 매칭
    links = {l["pred_track_uid"]: l for l in link_tracks(pred, gt)}
    assert links["P1"]["gt_track_uid"] == "G1" and links["P1"]["n_matched"] == 5
    assert links["P2"]["linked"] is False and links["P2"]["n_matched"] == 2
    assert links["P3"]["gt_track_uid"] is None


def test_link_one_to_one_per_frame():
    """두 pred가 같은 GT와 겹치면 IoU 높은 pred만 그 프레임에서 배정."""
    gt = [_obs("G1", "v", t, [100, 100, 200, 200]) for t in range(0, 1000, 200)]
    pa = [_obs("PA", "v", t, [100, 100, 200, 200]) for t in range(0, 1000, 200)]
    pb = [_obs("PB", "v", t, [110, 100, 210, 200]) for t in range(0, 1000, 200)]
    links = {l["pred_track_uid"]: l for l in link_tracks(pa + pb, gt)}
    assert links["PA"]["linked"] and links["PA"]["n_matched"] == 5
    assert not links["PB"]["linked"] and links["PB"]["n_matched"] == 0


def test_link_ratio_rule():
    """매칭 3프레임이어도 판정 가능 관측 대비 비율 < 0.5면 미연결."""
    gt = [_obs("G1", "v", t, [100, 100, 200, 200]) for t in range(0, 2000, 200)]
    pred = [_obs("P", "v", t, [100, 100, 200, 200] if t < 600 else [600, 600, 700, 700])
            for t in range(0, 2000, 200)]
    l = link_tracks(pred, gt)[0]
    assert l["n_matched"] == 3 and l["n_judgeable"] == 10 and not l["linked"]


# ---------------------------------------------------------------- 보존
MAN = {"v": {"width": 1920, "height": 1080, "duration_ms": 10000}}


def test_scene_visibility_center_crop():
    # 1920x1080 → 짧은 변 224 기준 center crop: 원본 x 420~1500 만 남음
    ok, _ = scene_visible([800, 400, 900, 600], 1920, 1080, CFG)
    assert ok
    ok, why = scene_visible([0, 400, 300, 600], 1920, 1080, CFG)          # 왼쪽 가장자리 → 잘림
    assert not ok and why == "cropped_out"
    ok, why = scene_visible([900, 500, 920, 520], 1920, 1080, CFG)        # 20px → 입력 약 4px
    assert not ok and why == "too_small"


def test_object_visibility():
    ok, _ = object_visible([100, 100, 200, 300], [100, 100, 200, 300], 1920, 1080, CFG)
    assert ok
    ok, why = object_visible([100, 100, 200, 300], [300, 100, 400, 300], 1920, 1080, CFG)
    assert not ok and why == "iou"


def test_retention_classification():
    gt = [_obs("G1", "v", t, [800, 400, 900, 600]) for t in range(0, 2000, 200)]
    gt += [_obs("G2", "v", t, [850, 450, 950, 650]) for t in range(0, 2000, 200)]
    gt += [_obs("G3", "v", t, [800, 700, 900, 900]) for t in range(0, 2000, 200)]
    targets = [
        {"target_id": "t1", "target_type": "track", "gt_track_uids": ["G1"], "video_id": "v", "split": "dev",
         "start_ms": 0, "end_ms": 2000, "evaluable": True},
        {"target_id": "t2", "target_type": "track", "gt_track_uids": ["G2"], "video_id": "v", "split": "dev",
         "start_ms": 0, "end_ms": 2000, "evaluable": True},
        {"target_id": "t3", "target_type": "track", "gt_track_uids": ["G3"], "video_id": "v", "split": "dev",
         "start_ms": 0, "end_ms": 2000, "evaluable": True},
        {"target_id": "t4", "target_type": "track", "gt_track_uids": ["G9"], "video_id": "v", "split": "dev",
         "start_ms": 0, "end_ms": 2000, "evaluable": False},
    ]
    samples = [
        {"item_id": "o1", "video_id": "v", "timestamp_ms": 400, "representation": "object",
         "track_uid": "x", "bbox": [800, 400, 900, 600]},                       # G1 유효, 인코딩됨
        {"item_id": "o2", "video_id": "v", "timestamp_ms": 600, "representation": "object",
         "track_uid": "y", "bbox": [850, 450, 950, 650]},                       # G2 유효, 인코딩 실패
    ]
    res = {r["target_id"]: r for r in classify_targets(targets, samples, gt, MAN, CFG, {"o1"},
                                                         representations=("object",))}
    assert res["t1"]["status"] == "indexed"
    assert res["t2"]["reason"] == "encoding_missing"
    assert res["t3"]["reason"] == "sampling_missing"
    assert res["t4"]["status"] == "N/A"
    s = summarize(list(res.values()))["track"]
    assert s["total_targets"] == 3 and s["indexed"] == 1 and math.isclose(s["missing_rate"], 2 / 3)


def test_activity_three_parts():
    gt = [_obs("P", "v", t, [800, 400, 900, 600]) for t in range(0, 3000, 200)]
    tgt = [{"target_id": "a1", "target_type": "activity", "gt_track_uids": ["P"],
            "participants": [{"track_uid": "P", "required": True}], "video_id": "v", "split": "dev",
            "start_ms": 0, "end_ms": 3000, "evaluable": True}]
    mk = lambda t: {"item_id": f"s{t}", "video_id": "v", "timestamp_ms": t, "representation": "scene",
                    "track_uid": None, "bbox": None}
    ok = classify_targets(tgt, [mk(200), mk(1200), mk(2200)], gt, MAN, CFG, None)[0]
    assert ok["status"] == "indexed"
    miss = classify_targets(tgt, [mk(200), mk(400), mk(2200)], gt, MAN, CFG, None)[0]  # 중간 부분 없음
    assert miss["status"] == "missing" and miss["reason"] == "sampling_missing"


def test_detection_missing_in_pred_mode():
    gt = [_obs("G1", "v", t, [800, 400, 900, 600]) for t in range(0, 2000, 200)]
    tgt = [{"target_id": "t1", "target_type": "track", "gt_track_uids": ["G1"], "video_id": "v",
            "split": "dev", "start_ms": 0, "end_ms": 2000, "evaluable": True}]
    r = classify_targets(tgt, [], gt, MAN, CFG, None, pred_to_gt={"p1": "G7"})[0]
    assert r["reason"] == "detection_missing"


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    sys.exit(1 if fails else 0)
