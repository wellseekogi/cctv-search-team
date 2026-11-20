"""Fail-closed readiness check before real B/C media integration.

Missing metadata is reported as pending; an empty error log never means completed.
After official inputs arrive, sampling must be rerun with those inputs first.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def validate_manifest(video):
    if video.get('provisional') or video.get('manifest_kind')=='observation_envelope_NOT_MEDIA_METADATA':
        raise ValueError('Provisional observation envelope cannot be used as media metadata')
    if video.get('split') not in ('dev','eval'):
        raise ValueError('F must confirm dev/eval split')
    if not isinstance(video.get('relative_path'),str) or not video['relative_path']:
        raise ValueError('Original media relative_path required')
    for key in ('width','height','duration_ms','available_start_ms','available_end_ms','source_offset_ms'):
        if type(video.get(key)) is not int or video[key]<0:
            raise ValueError('Invalid '+key)
    if not (video['width']>0 and video['height']>0 and video['source_offset_ms']<=video['available_start_ms']<video['available_end_ms']<=video['duration_ms']):
        raise ValueError('Invalid media dimensions or interval')
    if video.get('schema_version')!='cctv-3.0':
        raise ValueError('Incompatible schema')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--manifest',type=Path)
    p.add_argument('--video-status',type=Path)
    p.add_argument('--query-exclusions',type=Path)
    p.add_argument('--query-exclusions-confirmed',action='store_true')
    p.add_argument('--data-root',type=Path)
    p.add_argument('--frame-reader',help='configured module:function; check one scene/object pair per camera')
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.report.exists():
        raise FileExistsError('Use a fresh report path')
    issues=[]
    checks=[]
    inputs=list((args.run/'inputs').iterdir())
    videos={}
    if not args.manifest:
        issues.append('Official manifest not provided')
    else:
        ms=load(args.manifest)
        videos={m['video_id']:m for m in ms}
        if len(videos)!=len(ms):issues.append('Duplicate video IDs in official manifest')
        for folder in inputs:
            frames=load(folder/'frames.jsonl')
            try:
                m=videos[frames[0]['video_id']]
                validate_manifest(m)
                for f in frames:
                    if not (m['available_start_ms']<=f['timestamp_ms']<m['available_end_ms']):
                        raise ValueError('Supplied frame outside official interval; reconcile with B/F before rerun')
                    if (f['width'],f['height'],f['source_offset_ms'])!=(m['width'],m['height'],m['source_offset_ms']):
                        raise ValueError('B/manifest dimensions or offset mismatch')
                recorded=folder/'manifest.jsonl'
                if not recorded.exists():recorded=folder/'manifest.PROVISIONAL.jsonl'
                stub=load(recorded)[0]
                if m['file_sha256']!=stub['file_sha256']:
                    raise ValueError('Official hash differs from B reported original hash')
                if args.data_root:
                    source=(args.data_root/m['relative_path']).resolve()
                    if not source.is_relative_to(args.data_root.resolve()):raise ValueError('Media path escapes root')
                    with source.open('rb') as f:
                        digest=hashlib.file_digest(f,'sha256').hexdigest()
                    if digest!=m['file_sha256']:raise ValueError('Original file SHA256 mismatch')
                checks.append(folder.name+': official metadata PASS')
            except (KeyError,ValueError,OSError) as e:issues.append(folder.name+': '+str(e))
    if not args.data_root:issues.append('Original video data_root not provided')
    if not args.video_status:issues.append('Explicit video completion status not provided')
    else:
        statuses={s['video_id']:s for s in load(args.video_status)}
        for folder in inputs:
            vid=load(folder/'frames.jsonl')[0]['video_id']
            s=statuses.get(vid,{})
            if s.get('video_status')!='completed' or s.get('available_end_ms')!=videos.get(vid,{}).get('available_end_ms'):
                issues.append(folder.name+': completion missing/mismatched')
    if not args.query_exclusions or not args.query_exclusions_confirmed:
        issues.append('F-confirmed query exclusions required, including explicit empty confirmation')
    else:
        for row in load(args.query_exclusions):
            m=videos.get(row.get('video_id'),{})
            a,b=row.get('start_ms'),row.get('end_ms')
            if type(a)is not int or type(b)is not int or not 0<=a<b<=m.get('duration_ms',-1):
                issues.append('Invalid query exclusion row')
    if not args.frame_reader:
        issues.append('Current B reader and pixel PTS/frame smoke check pending')
    elif not issues:
        from integrations.b_reader import motion_frame
        import numpy as np
        mod,fun=args.frame_reader.split(':',1)
        reader=getattr(importlib.import_module(mod),fun)
        for folder in inputs:
            smoke=load(args.run/'results'/folder.name/'object_event/samples_smoke.jsonl')
            fs={f['frame_index']:f for f in load(folder/'frames.jsonl')}
            try:
                for index in {s['frame_index'] for s in smoke}:
                    f=fs[index]
                    rgb=np.asarray(motion_frame(reader,f))
                    if rgb.dtype!=np.uint8 or rgb.shape!=(f['height'],f['width'],3):
                        raise ValueError('Invalid RGB dtype/shape')
                checks.append(folder.name+': B pixel metadata smoke PASS (not full-video verification)')
            except Exception as e:issues.append(folder.name+': '+str(e))
    result=dict(status='PENDING' if issues else 'READY_TO_RERUN_WITH_OFFICIAL_INPUTS',
        checks=checks,remaining=issues,existing_partial_results_final=False,
        next='Rerun A on official interval/split/exclusions; then C build + validate and F/G acceptance. Do not relabel partial results as final.')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True,indent=2))
    return 2 if issues else 0


if __name__=='__main__':sys.exit(main())
