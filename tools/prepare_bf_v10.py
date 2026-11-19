"""Rebuild A results using pinned F metadata and unchanged B observation IDs."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.validate_a_output import rows, sha, c_checks, verify_output
from sampling.core import run_sampling
from sampling.__main__ import write_json, write_jsonl
from integrations.f_metadata import align_manifest, image_exclusions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--b-inputs',type=Path,required=True,help='Directory of B per-camera inputs from v8/v10 delivery')
    p.add_argument('--f-data',type=Path,required=True,help='F checkout data/meva directory')
    args=p.parse_args()
    if args.out.exists():raise FileExistsError('Use a fresh output path')
    ms=rows((args.f_data/'manifest.jsonl').read_bytes())
    queries=rows((args.f_data/'queries.jsonl').read_bytes())
    input_dirs=sorted(p for p in args.b_inputs.iterdir() if p.is_dir())
    if not input_dirs:raise ValueError('No B per-camera input folders found')
    if not ms:raise ValueError('F manifest is empty')
    args.out.mkdir(parents=True)
    checks=c_checks()
    summary=[]
    mapping=[]
    all_manifests=[]
    all_exclusions=[]
    for old in input_dirs:
        camera=old.name
        frames=rows((old/'frames.jsonl').read_bytes())
        tracks=rows((old/'tracks.jsonl').read_bytes())
        manifest_path=old/'manifest.PROVISIONAL.jsonl'
        if not manifest_path.exists():manifest_path=old/'manifest.jsonl'
        b=rows(manifest_path.read_bytes())[0]
        manifest=align_manifest(b,ms,frames)
        exclusions=image_exclusions(queries,manifest)
        all_manifests.append(manifest)
        all_exclusions.extend(exclusions)
        mapping.append(dict(b_video_id=b['video_id'],f_video_id=manifest['f_video_id'],
                            file_sha256=manifest['file_sha256'],split=manifest['split'],
                            camera=camera,exclusions=len(exclusions),match_basis='reported_original_sha256'))
        statuses=[dict(video_id=b['video_id'],video_status='completion_pending',available_end_ms=manifest['available_end_ms'])]
        inputs=[[manifest],frames,tracks,statuses,exclusions]
        folder=args.out/'inputs'/camera
        folder.mkdir(parents=True)
        for name,data in zip(('manifest.jsonl','frames.jsonl','tracks.jsonl','video_status.jsonl','query_exclusions.jsonl'),inputs):
            write_jsonl(folder/name,data)
        base=json.loads((ROOT/'configs/person/sampling.yaml').read_text())
        base.update(execution_mode='observed_partial',query_exclusion_status='F_queries_embedded_exclusions',
            supported_classes=['person'],track_source_id=tracks[0]['track_source_id'],
            manifest_status='F_git_hash_matched_B_alias',
            observation_identity=dict(version='BF_v10',frames_sha256=sha((old/'frames.jsonl').read_bytes()),
                tracks_sha256=sha((old/'tracks.jsonl').read_bytes()),
                F_manifest_sha256=sha((args.f_data/'manifest.jsonl').read_bytes()),
                F_queries_sha256=sha((args.f_data/'queries.jsonl').read_bytes())))
        for policy,step in (('object_event',None),('hybrid',None),('uniform',1000),('uniform',2000),('uniform',5000)):
            cfg=copy.deepcopy(base)
            cfg['sampling']['policy']=policy
            if step:cfg['sampling']['fixed_step_ms']=step
            name=policy if step is None else 'uniform_'+str(step)
            run_id=f'meva-t3-{camera}-{name}-partial-v10'
            events,samples,info=run_sampling(cfg,*inputs,run_id=run_id)
            assert (events,samples,info)==run_sampling(cfg,[manifest],list(reversed(frames)),list(reversed(tracks)),statuses,exclusions,run_id=run_id)
            assert all(not any(x['start_ms'] <= s['timestamp_ms'] < x['end_ms'] for x in exclusions) for s in samples)
            crops=verify_output(events,samples,info,frames,tracks,manifest,cfg,checks)
            result=args.out/'results'/camera/name
            result.mkdir(parents=True)
            info.update(manifest_status='F_git_hash_matched_B_alias',query_exclusion_status='F_queries_embedded_exclusions',
                c_media_ready=False,final_evaluation_ready=False,split=manifest['split'])
            for filename,data in [('events.jsonl',events),('samples.jsonl',samples),('errors.jsonl',[])]:
                write_jsonl(result/filename,data)
            write_json(result/'sampling.yaml',cfg)
            write_json(result/'selection_summary.json',info)
            write_json(result/'stage_status.json',dict(status='partial_success',c_media_ready=False,final_evaluation_ready=False))
            scene_frames={s['frame_index'] for s in samples if s['representation']=='scene'}
            obj=next((s for s in samples if s['representation']=='object' and s['frame_index'] in scene_frames),None)
            smoke=[] if obj is None else [s for s in samples if s['item_id']==obj['item_id'] or (s['representation']=='scene' and s['frame_index']==obj['frame_index'])]
            write_jsonl(result/'samples_smoke.jsonl',smoke)
            write_jsonl(result/'expected_smoke_crop.jsonl',[c for c in crops if obj and c['item_id']==obj['item_id']])
            record=dict(camera=camera,policy=name,counts=info['counts'],event_types=info['event_types'],
                deterministic='PASS',B_frame_track_identity='PASS',C_sample_and_crop_contract='PASS',
                event_references='PASS',intervals_caps='PASS',final_evaluation_ready=False)
            write_json(result/'verification.json',record)
            summary.append(record)
            print(camera,name,info['counts'],flush=True)
        motion=copy.deepcopy(base)
        motion['sampling']['policy']='motion'
        write_json(folder/'motion_PENDING.yaml',motion)
    write_json(args.out/'video_id_map.json',mapping)
    write_jsonl(args.out/'manifest.B_ALIASED.jsonl',all_manifests)
    write_jsonl(args.out/'query_exclusions.jsonl',all_exclusions)
    write_json(args.out/'integration_summary.json',dict(status='partial_success',runs=summary,
        F_image_query_exclusions=len(all_exclusions),G330='not_in_F_manifest; retain v9 example separately',
        video_id_strategy='B IDs preserved; map to F by reported original SHA256; never map pred IDs to GT IDs',
        blocked=['B explicit completion','original video and pixel/PTS validation','F GT observations and query images',
                 'motion, C embeddings and F/G independent acceptance']))


if __name__=='__main__':main()
