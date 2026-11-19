import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class BFCLITests(unittest.TestCase):
    def test_external_inputs_exclude_query_and_keep_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            b=root/'input'/'camera';b.mkdir(parents=True)
            f=root/'F';f.mkdir()
            def write(path, rows):
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            manifest=dict(schema_version='cctv-3.0',video_id='b_video',file_sha256='a'*64,
                width=100,height=100,source_offset_ms=0,duration_ms=1401,
                available_start_ms=0,available_end_ms=1401,frame_count=7,split='dev',relative_path='video.avi')
            write(b/'manifest.jsonl',[manifest])
            write(f/'manifest.jsonl',[dict(manifest,video_id='f_video')])
            frames=[dict(schema_version='cctv-3.0',video_id='b_video',frame_index=i,timestamp_ms=t,
                requested_timestamp_ms=t,width=100,height=100,source_offset_ms=0,
                time_source='pts',decode_status='success',detector_status='ran') for i,t in enumerate([0,200,400,600,800,1000,1200])]
            write(b/'frames.jsonl',frames)
            tracks=[dict(schema_version='cctv-3.0',video_id='b_video',frame_index=x['frame_index'],timestamp_ms=x['timestamp_ms'],
                source='pred',track_source_id='litedetr01',local_track_id='t1',track_uid='litedetr01:b_video:t1',
                class_name='person',bbox=[10,10,30,40],confidence=.9,quality=None) for x in frames]
            write(b/'tracks.jsonl',tracks)
            write(f/'queries.jsonl',[dict(query_id='image1',modality='image',split='dev',
                source=dict(video_id='f_video'),exclusion=dict(video_id='f_video',start_ms=0,end_ms=200,
                applies_to=['scene','object'],all_policies=True))])
            cmd=[sys.executable,str(ROOT/'tools/prepare_bf_v10.py'),'--b-inputs',str(b.parent),'--f-data',str(f),'--out',str(root/'out')]
            proc=subprocess.run(cmd,cwd=temp,capture_output=True,text=True)
            self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
            paths=list((root/'out/results').glob('*/*/samples.jsonl'))
            self.assertEqual(len(paths),5)
            samples=[json.loads(line) for p in paths for line in p.read_text().splitlines()]
            self.assertTrue(samples)
            self.assertTrue(all(s['timestamp_ms']>=200 and s['video_id']=='b_video' for s in samples))
            self.assertTrue(all(s['track_uid']=='litedetr01:b_video:t1' for s in samples if s['representation']=='object'))
            self.assertNotEqual(subprocess.run(cmd,cwd=temp,capture_output=True).returncode,0)
