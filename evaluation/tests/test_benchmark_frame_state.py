import unittest

import numpy as np

from evaluation.benchmark_frame_state import proposal_metrics, rank_result, tiou
from team_search.frame_state import rank_windows


class FrameStateMetricTests(unittest.TestCase):
    def test_rank_separates_center_and_half_open_iou_in_the_correct_camera(self):
        truth = [dict(video_id="plaza", start_ms=1000, end_ms=3000)]
        ranked = [
            dict(video_id="gym", start_ms=1000, end_ms=3000, timestamp_ms=2000),
            dict(video_id="plaza", start_ms=0, end_ms=4000, timestamp_ms=3000),
            dict(video_id="plaza", start_ms=1400, end_ms=1600, timestamp_ms=1500),
        ]
        self.assertEqual(tiou(ranked[1], truth[0]), 0.5)
        self.assertEqual(rank_result(ranked, truth), {"rank": 3, "interval_rank_50": 2})
        self.assertEqual(rank_result(ranked[:1], truth), {"rank": None, "interval_rank_50": None})

    def test_proposals_count_each_activity_once_and_keep_uncovered_activities(self):
        activities = [dict(video_id="plaza", start_ms=0, end_ms=4000),
                      dict(video_id="plaza", start_ms=2000, end_ms=3000),
                      dict(video_id="gym", start_ms=0, end_ms=1000)]
        proposal = dict(video_id="plaza", start_ms=0, end_ms=2000)
        self.assertEqual(proposal_metrics([proposal, proposal], activities), {
            "annotated_activities": 3, "overlap": 1, "tiou_30": 1, "tiou_50": 1, "tiou_70": 0,
        })

    def test_both_policies_rank_time_and_view_axes_identically(self):
        # A scores +0.5 with three frames and -0.5 with its center. B scores 0 and +1.
        positive, negative, other = [1, 0], [-1, 0], [0, 1]
        vectors = np.array([[[positive, positive], [negative, other], [positive, positive]],
                            [[other, other], [positive, positive], [negative, negative]]], dtype=np.float32)
        for source in ("change", "uniform"):
            rows = [dict(video_id="A", source=source), dict(video_id="B", source=source)]
            full = rank_windows(rows, vectors, positive)
            center = rank_windows(rows, vectors, positive, center_only=True)
            self.assertEqual([(r["video_id"], r["similarity"]) for r in full], [("A", .5), ("B", 0)])
            self.assertEqual([(r["video_id"], r["similarity"]) for r in center], [("B", 1), ("A", -.5)])
            self.assertEqual(rank_windows(rows, vectors[:, ::-1], positive), full)
