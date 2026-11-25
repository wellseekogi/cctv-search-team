import copy
import json
import unittest
from pathlib import Path

import numpy as np
from sampling.__main__ import read_jsonl
from sampling.core import InputError, run_sampling

ROOT = Path(__file__).resolve().parents[1]


class PersonScopeTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/sampling/person/sampling.yaml').read_text())
        self.inputs = [read_jsonl(ROOT/'fixtures/event_trace'/n) for n in
                       ('manifest.jsonl', 'frames.jsonl', 'tracks.jsonl',
                        'video_status.jsonl', 'query_exclusions.jsonl')]
        people = copy.deepcopy(self.inputs[2])
        for row in people:
            row.update(class_name='person', local_track_id='person1')
            row['track_uid'] = f"{row['track_source_id']}:{row['video_id']}:person1"
        self.inputs[2].extend(people)

    def run_case(self):
        return run_sampling(self.config, *self.inputs, frame_loader=lambda f:
                            np.zeros((f['height'], f['width'], 3), dtype=np.uint8))

    def test_all_policies_exclude_vehicle_events_and_objects(self):
        for policy in ('object_event', 'hybrid', 'uniform', 'motion'):
            with self.subTest(policy=policy):
                self.config['sampling']['policy'] = policy
                events, samples, summary = self.run_case()
                self.assertTrue(samples)
                self.assertTrue(all(e['track_uid'].endswith(':person1') for e in events))
                objects = [s for s in samples if s['representation']=='object']
                self.assertTrue(objects)
                self.assertTrue(all(s['track_uid'].endswith(':person1') for s in objects))
                self.assertGreater(summary['ignored_observations_by_class']['car'], 0)
                self.assertFalse(summary['action_recognition_implemented'])

    def test_vehicle_only_has_no_object_events_but_periodic_scene_survives(self):
        self.inputs[2] = [r for r in self.inputs[2] if r['class_name']=='car']
        for policy in ('object_event', 'hybrid', 'uniform', 'motion'):
            self.config['sampling']['policy'] = policy
            events, samples, summary = self.run_case()
            self.assertEqual(events, [])
            self.assertTrue(all(s['representation']=='scene' for s in samples))
            self.assertEqual(summary['target_observations'], 0)
            self.assertEqual(bool(samples), policy != 'object_event')

    def test_filter_does_not_hide_invalid_vehicle_input(self):
        self.inputs[2][0]['bbox'] = [-1, 0, 1, 1]
        with self.assertRaisesRegex(InputError, 'bbox out of bounds'):
            self.run_case()

    def test_person_filter_matches_manually_filtered_input_and_is_deterministic(self):
        result = self.run_case()
        self.inputs[2].reverse()
        self.assertEqual(result, self.run_case())
        self.inputs[2] = [r for r in self.inputs[2] if r['class_name']=='person']
        manual = self.run_case()
        self.assertEqual(result[:2], manual[:2])

    def test_invalid_target_classes_fail(self):
        for target in ([], ['unknown'], 'person', ['person', 'person']):
            self.config['target_classes'] = target
            with self.assertRaisesRegex(InputError, 'target_classes'):
                self.run_case()
