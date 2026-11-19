import unittest
from integrations.f_metadata import align_manifest, image_exclusions


class FMetadataTests(unittest.TestCase):
    def setUp(self):
        self.b = dict(video_id='meva_clip', file_sha256='abc', width=20, height=10, source_offset_ms=0)
        self.f = dict(self.b, video_id='clip', split='dev', duration_ms=1000,
                      available_start_ms=0, available_end_ms=1000, frame_count=30)
        self.frames = [dict(video_id='meva_clip', timestamp_ms=200, frame_index=6)]

    def test_hash_match_keeps_b_namespace(self):
        m=align_manifest(self.b,[self.f],self.frames)
        self.assertEqual((m['video_id'],m['f_video_id']),('meva_clip','clip'))

    def test_wrong_or_ambiguous_hash_rejected(self):
        for ms in ([dict(self.f,file_sha256='wrong')],[self.f,self.f]):
            with self.assertRaises(ValueError):align_manifest(self.b,ms,self.frames)

    def test_exclusive_media_end(self):
        with self.assertRaises(ValueError):
            align_manifest(self.b,[self.f],[dict(self.frames[0],timestamp_ms=1000)])

    def test_query_alias_and_split_guard(self):
        m=align_manifest(self.b,[self.f],self.frames)
        q=dict(query_id='q',modality='image',split='dev',source=dict(video_id='clip'),
               exclusion=dict(video_id='clip',start_ms=100,end_ms=300,
                              applies_to=['scene','object'],all_policies=True))
        self.assertEqual(image_exclusions([q],m)[0]['video_id'],'meva_clip')
        q['split']='eval'
        with self.assertRaises(ValueError):image_exclusions([q],m)
