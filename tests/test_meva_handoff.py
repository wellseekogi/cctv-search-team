import copy
import unittest
from tools.check_meva_ready import validate_manifest
from integrations.b_reader import normalize_frame


class MevaHandoffTests(unittest.TestCase):
    def setUp(self):
        self.m=dict(schema_version='cctv-3.0',split='dev',relative_path='x.avi',width=1920,height=1080,
                    duration_ms=300001,available_start_ms=0,available_end_ms=300001,source_offset_ms=0)

    def test_provisional_manifest_is_rejected(self):
        for extra in ({'provisional':True},{'manifest_kind':'observation_envelope_NOT_MEDIA_METADATA'},
                      {'split':'unassigned'},{'relative_path':None}):
            with self.subTest(extra=extra),self.assertRaises(ValueError):
                validate_manifest(dict(self.m,**extra))

    def test_real_end_is_exclusive(self):
        m=copy.deepcopy(self.m)
        m['available_start_ms']=m['available_end_ms']
        with self.assertRaises(ValueError):validate_manifest(m)
        validate_manifest(self.m)

    def test_b_c_nearest_time_is_not_exact_selected_frame(self):
        frame=dict(rgb='unused',actual_timestamp_ms=233,frame_index=6,width=1920,height=1080,
                   source_offset_ms=0,time_source='pts')
        with self.assertRaisesRegex(ValueError,'FRAME_MISMATCH'):normalize_frame(frame,200,6)
        frame['actual_timestamp_ms']=200
        with self.assertRaisesRegex(ValueError,'FRAME_MISMATCH'):normalize_frame(frame,200,7)
        self.assertEqual(normalize_frame(frame,200,6)['image_rgb'],'unused')
