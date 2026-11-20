import json
import unittest
from pathlib import Path
import numpy as np
from sampling.__main__ import read_jsonl
from sampling.core import InputError,run_sampling
from sampling.motion import gray_frame

ROOT=Path(__file__).resolve().parents[1]

class MotionTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((ROOT/"configs/sampling.yaml").read_text(encoding="utf-8"))
        self.config["sampling"]["policy"]="motion"
        self.inputs=[read_jsonl(ROOT/"fixtures/event_trace"/n) for n in
                    ("manifest.jsonl","frames.jsonl","tracks.jsonl","video_status.jsonl","query_exclusions.jsonl")]
        self.inputs[0][0].update(width=160,height=90)
        self.inputs[2]=[]
        self.inputs[1]=[dict(f,width=160,height=90) for f in self.inputs[1][:4]]
        self.levels=[0,7,14,22]

    def loader(self,f):
        return np.full((90,160,3),self.levels[f["timestamp_ms"]//200],np.uint8)

    def run_case(self):
        return run_sampling(self.config,*self.inputs,frame_loader=self.loader)

    def test_consecutive_not_last_selected(self):
        _,samples,summary=self.run_case()
        self.assertEqual([s["timestamp_ms"] for s in samples],[0,600])
        self.assertAlmostEqual(summary["motion_scores"][2]["motion_score"],7/255)
        self.assertTrue(all(s["reasons"]==["motion"] and s["event_ids"]==[] for s in samples))

    def test_float_difference_not_unsigned_wrap(self):
        self.levels=[255,0,0,0]
        self.assertEqual(self.run_case()[2]["motion_scores"][1]["motion_score"],1)

    def test_threshold_inclusive(self):
        self.config["sampling"]["motion_threshold"]=7/255
        self.assertEqual(len(self.run_case()[1]),4)

    def test_excluded_frame_still_updates_comparison(self):
        self.levels=[0,255,255,255]
        self.inputs[4]=[{"video_id":"v001","start_ms":0,"end_ms":400}]
        self.assertEqual(self.run_case()[1],[])

    def test_known_failure_resets_without_loading_failed_frame(self):
        self.inputs[1][1]["decode_status"]="failed"
        self.assertEqual([s["timestamp_ms"] for s in self.run_case()[1]],[0,400,600])
        self.assertEqual(self.run_case()[2]["motion_scores"][1]["status"],"DECODE_FAILED")

    def test_missing_loader_rejected(self):
        with self.assertRaises(InputError):
            run_sampling(self.config,*self.inputs)

    def test_bad_dimensions_rejected(self):
        with self.assertRaises(InputError):
            run_sampling(self.config,*self.inputs,frame_loader=lambda f:np.zeros((1,1,3),np.uint8))

    def test_deterministic_and_scene_cap(self):
        first=self.run_case()
        self.inputs[1].reverse()
        self.assertEqual(first,self.run_case())
        self.config["sampling"]["max_scene_per_minute"]=1
        self.assertEqual(self.run_case()[2]["counts"]["scene"],1)

    def test_grayscale_resize_contract(self):
        f=dict(width=320,height=180)
        result=gray_frame(np.full((180,320,3),8,np.uint8),f)
        self.assertEqual(result.shape,(90,160))
        self.assertEqual(result.dtype,np.float32)
        self.assertTrue(np.all(result==8))

    def test_objects_use_shared_interval(self):
        o=dict(schema_version="cctv-3.0",source="gt",track_source_id="gt",video_id="v001",
               local_track_id="car",track_uid="gt:v001:car",class_name="car",confidence=None,
               bbox=[20,20,60,60],quality=None)
        self.levels=[0,255,0,255]
        self.inputs[2]=[dict(o,frame_index=f["frame_index"],timestamp_ms=f["timestamp_ms"]) for f in self.inputs[1]]
        samples=self.run_case()[1]
        self.assertEqual([s["timestamp_ms"] for s in samples if s["representation"]=="object"],[0,600])
        self.assertEqual(len([s for s in samples if s["representation"]=="scene"]),4)

    def test_new_video_resets_comparison(self):
        self.inputs[0].append(dict(self.inputs[0][0],video_id="v002"))
        self.inputs[3].append(dict(self.inputs[3][0],video_id="v002"))
        self.inputs[1].append(dict(self.inputs[1][0],video_id="v002"))
        self.assertEqual(self.run_case()[2]["motion_scores"][-1]["status"],"FIRST_VALID")
