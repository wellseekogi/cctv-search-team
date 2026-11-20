import unittest

from evaluation.benchmark_meva_search import (
    localized_person,
    rank_stats,
    relevant_frame,
)
from evaluation.summarize_web_search import compare, time_verdict


class SearchAuditTests(unittest.TestCase):
    def test_timestamp_does_not_credit_the_wrong_person(self):
        truth = [{"video_id": "clip", "start_ms": 200, "end_ms": 400, "gt_track_uid": "person-1"}]
        hit = {"video_id": "meva_clip", "timestamp_ms": 200, "frame_index": 6, "bbox": [0, 0, 10, 10]}
        boxes = {("clip", 6): {"person-1": [5, 0, 15, 10]}}
        self.assertTrue(relevant_frame(hit, truth))
        self.assertFalse(localized_person(hit, truth, boxes))
        boxes["clip", 6]["person-1"] = [0, 0, 5, 10]
        self.assertTrue(localized_person(hit, truth, boxes))  # IoU exactly 0.5.
        self.assertFalse(localized_person(dict(hit, bbox=None), truth, boxes))
        self.assertFalse(relevant_frame(dict(hit, timestamp_ms=400), truth))
        self.assertFalse(relevant_frame(dict(hit, video_id="other"), truth))
        self.assertEqual(rank_stats([False, True, True]), {"rank": 2, "relevant": 2, "candidates": 3})

    def test_follow_up_uses_the_previous_result_and_preserves_errors(self):
        hit = {"item_id": "current", "score": 0.1, "playback_time_ms": 9800}
        row = {
            "query": "q",
            "status": "generated",
            "results": [hit, hit],
            "answer": "9.8초부터",
            "previous_second": {"item_id": "previous", "playback_time_ms": 7200},
        }
        result = compare([row, {"query": "q", "status": "error"}])["q"]
        self.assertEqual((result["requests"], result["generated"]), (2, 1))
        self.assertEqual(result["answer_time"], {"wrong": 1})
        self.assertEqual(result["previous_second_still_second"], 0)
        self.assertEqual(time_verdict("9.8 초", 9833), "matched")
        self.assertEqual(time_verdict("바로 확인하세요", 0), "unspecified")
        self.assertEqual(time_verdict("9.8초 또는 8초", 9833), "wrong")


if __name__ == "__main__":
    unittest.main()
