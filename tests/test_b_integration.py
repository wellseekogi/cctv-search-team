import json
from pathlib import Path
import unittest
import numpy as np
from integrations.b_reader import normalize_frame, motion_frame
from sampling.core import run_sampling, InputError
from sampling.__main__ import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


class BIntegrationTests(unittest.TestCase):
    def test_rgb_key_adapter(self):
        rgb = np.zeros((90,160,3),np.uint8)
        frame = dict(rgb=rgb, actual_timestamp_ms=200, frame_index=2,width=160,height=90,source_offset_ms=0,time_source="pts")
        mapped = normalize_frame(frame,200,2)
        self.assertIs(mapped["image_rgb"],rgb)
        self.assertNotIn("image_rgb",frame)
        self.assertIs(motion_frame(lambda *args:frame,dict(video_id="v",timestamp_ms=200,frame_index=2,width=160,height=90)),rgb)

    def test_adapter_rejects_nearby_wrong_frame(self):
        frame = dict(rgb=None,actual_timestamp_ms=220,frame_index=2,width=160,height=90,source_offset_ms=0,time_source="pts")
        with self.assertRaisesRegex(ValueError,"FRAME_MISMATCH"):
            normalize_frame(frame,200)
        frame["actual_timestamp_ms"] = 200
        with self.assertRaisesRegex(ValueError,"FRAME_MISMATCH"):
            normalize_frame(frame,200,3)

    def test_partial_enables_gain_gap_resume_not_end(self):
        cfg = json.loads((ROOT/"configs/sampling.yaml").read_text(encoding="utf-8"))
        cfg["execution_mode"] = "observed_partial"
        inputs = [read_jsonl(ROOT/"fixtures/event_trace"/n) for n in
                  ("manifest.jsonl","frames.jsonl","tracks.jsonl","video_status.jsonl","query_exclusions.jsonl")]
        inputs[3][0]["video_status"] = "completion_pending"
        events,samples,summary = run_sampling(cfg,*inputs)
        kinds={e["event_type"] for e in events}
        self.assertTrue({"QUALITY_GAIN","TRACK_GAP","TRACK_RESUME"} <= kinds)
        self.assertNotIn("VIDEO_END",kinds)
        self.assertEqual(summary["counts"],dict(events=8,object=4,scene=7))
        self.assertFalse(summary["final_evaluation_ready"])
        self.assertTrue(all(s["provisional"] for s in samples))
        self.assertNotIn("CLOSED",summary["track_final_states"].values())
        cfg.pop("execution_mode")
        with self.assertRaises(InputError):
            run_sampling(cfg,*inputs)

    def test_partial_uniform_is_marked(self):
        cfg=json.loads((ROOT/"configs/uniform_2000.yaml").read_text(encoding="utf-8"))
        cfg["execution_mode"]="observed_partial"
        inputs=[read_jsonl(ROOT/"fixtures/event_trace"/n) for n in
                ("manifest.jsonl","frames.jsonl","tracks.jsonl","video_status.jsonl","query_exclusions.jsonl")]
        inputs[3][0]["video_status"]="completion_pending"
        _,samples,summary=run_sampling(cfg,*inputs)
        self.assertFalse(summary["final_evaluation_ready"])
        self.assertTrue(all(s["query_exclusion_status"]=="not_provided" for s in samples))
