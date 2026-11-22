import numpy as np

from evaluation.state_action_pairs import ACTION_PAIRS, evaluate_pairs


def test_opposite_actions_require_an_exclusive_label_and_score_ties_half():
    windows = [dict(video_id="cam", timestamp_ms=t) for t in (0, 1, 2, 3, 4)]
    up, down = [c for c, _ in ACTION_PAIRS[0]]
    targets = [dict(target_type="activity", evaluable=True, video_id="cam",
                    activity_name=label, start_ms=start, end_ms=end)
               for label, start, end in ((up, 0, 2), (down, 1, 4))]
    scores = np.zeros((5, 8))
    scores[0, 0] = 1
    scores[3, 0] = 1
    pair, missing, *_ = evaluate_pairs(windows, targets, {"ordered": scores})
    assert pair["indices"] == [0, 2, 3] and pair["counts"] == [1, 2]
    assert pair["simultaneous"] == 1
    assert pair["metrics"]["ordered"] == dict(credit=1.5, ties=1, recalls=[1.0, 0.25], balanced_accuracy=0.625)
    assert missing["metrics"]["ordered"]["balanced_accuracy"] is None
