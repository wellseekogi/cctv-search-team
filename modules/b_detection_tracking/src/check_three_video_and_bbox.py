#!/usr/bin/env python3
"""Non-destructive final-candidate QA for three Lite-DETR T3 CCTV exports.

Checks full frames/tracks/error schema, temporal grid, coordinates, quality values,
UID uniqueness across videos, and sample CSV; ranks bounding-box clamping and
renders top 10 severe comparisons for c007 and c034. Does not modify exports.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

EXPECTED = {
    'c007': ('cityflow_S02_c007',983),
    'c033': ('cityflow_S05_c033',1704),
    'c034': ('cityflow_S05_c034',1713),
}

def read_jsonl(path):
    with path.open(encoding='utf-8') as f:
        for line_no, line in enumerate(f,1):
            if line.strip():
                try: v=json.loads(line)
                except Exception as e: raise ValueError(f'{path}:{line_no}: {e}')
                if not isinstance(v,dict): raise ValueError(f'{path}:{line_no}: not object')
                yield v

def num(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def find_dirs(root):
    return {
        'c007':root/'runs/b_litedetr_multivideo/b3_b4_c007_t3',
        'c033':root/'runs/b_litedetr_dev_001/b3_b4_t3_candidate',
        'c034':root/'runs/b_litedetr_multivideo/b3_b4_c034_t3',
    }

def calc_removed(row):
    raw=row['bbox_original']; box=row['bbox_clamped']
    if len(raw)!=4 or len(box)!=4 or not all(num(x) for x in [*raw,*box]):
        raise ValueError('invalid bbox correction record')
    rw=max(0,raw[2]-raw[0]); rh=max(0,raw[3]-raw[1]); a=rw*rh
    cw=max(0,box[2]-box[0]); ch=max(0,box[3]-box[1]); b=cw*ch
    if a<=0 or b<=0: raise ValueError('empty corrected bbox')
    shift=max(abs(x-y) for x,y in zip(raw,box))
    pct=max(0,100*(1-b/a))
    sev='severe' if pct>=20 else ('moderate' if pct>=5 else 'minor')
    return pct,shift,sev

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,default=Path('runs/b_three_video_qa'))
    p.add_argument('--render-top',type=int,default=10)
    p.add_argument('--skip-images',action='store_true')
    args=p.parse_args()
    root=args.project.resolve(); out=(args.output if args.output.is_absolute() else root/args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    results={}; uid_owners={}; problems=[]; counts=Counter(); corrections={}
    for camera,d in find_dirs(root).items():
        vid,expected=EXPECTED[camera]
        for filename in ('frames.jsonl','tracks.jsonl','errors.jsonl','bbox_corrections.jsonl'):
            if not (d/filename).is_file(): raise FileNotFoundError(f'{camera}: {d/filename}')
        frames=list(read_jsonl(d/'frames.jsonl'))
        if len(frames)!=expected: problems.append(f'{camera}: frames count {len(frames)} != {expected}')
        times={}; shapes={}; seenframes=set(); last_actual=-1
        for i,r in enumerate(frames):
            if r.get('schema_version')!='cctv-3.0' or r.get('video_id')!=vid:
                problems.append(f'{camera}: frame schema/video id mismatch at row {i}')
            req=r.get('requested_timestamp_ms'); actual=r.get('timestamp_ms'); fi=r.get('frame_index')
            if req!=i*200 or not num(actual) or abs(actual-req)>100:
                problems.append(f'{camera}: frame pts mismatch at row {i}')
            if num(actual) and actual<=last_actual: problems.append(f'{camera}: nonascending pts row {i}')
            if num(actual):last_actual=actual
            if not isinstance(fi,int) or fi in seenframes: problems.append(f'{camera}: duplicate/invalid frame index at {i}')
            seenframes.add(fi)
            if r.get('decode_status')!='success' or r.get('detector_status')!='ran':
                problems.append(f'{camera}: frame decode/detect status bad at {i}')
            w,h=r.get('width'),r.get('height')
            if not isinstance(w,int) or not isinstance(h,int) or w<=0 or h<=0:
                problems.append(f'{camera}: bad dimensions at {i}')
            times[actual]=i; shapes[actual]=(w,h)
        unique=set(); observations=0; classes=Counter(); track_first={}
        for i,r in enumerate(read_jsonl(d/'tracks.jsonl')):
            observations+=1
            if r.get('schema_version')!='cctv-3.0' or r.get('video_id')!=vid or r.get('source')!='pred':
                problems.append(f'{camera}: track schema/video/source mismatch at observation {i}')
            ts=r.get('timestamp_ms')
            if ts not in times:
                problems.append(f'{camera}: track timestamp {ts} outside frames')
                continue
            if r.get('frame_index')!=frames[times[ts]].get('frame_index'):
                problems.append(f'{camera}: track frame_index wrong at observation {i}')
            uid=r.get('track_uid'); local=r.get('local_track_id'); sid=r.get('track_source_id')
            if not isinstance(uid,str) or not isinstance(sid,str) or not isinstance(local,str) or uid!=f'{sid}:{vid}:{local}':
                problems.append(f'{camera}: uid namespace mismatch {uid}')
            key=(uid,ts)
            if key in unique: problems.append(f'{camera}: duplicate (uid,ts) {key}')
            unique.add(key)
            if uid in track_first and ts<=track_first[uid]:
                problems.append(f'{camera}: unordered track {uid}')
            track_first[uid]=ts
            if uid in uid_owners and uid_owners[uid]!=camera:
                problems.append(f'{camera}: uid collision between videos {uid}')
            uid_owners[uid]=camera
            conf=r.get('confidence')
            if not num(conf) or not 0.25<=conf<=1:
                problems.append(f'{camera}: invalid confidence at {i}: {conf}')
            bbox=r.get('bbox');w,h=shapes[ts]
            if not isinstance(bbox,list) or len(bbox)!=4 or not all(num(x) for x in bbox) or not(0<=bbox[0]<bbox[2]<=w and 0<=bbox[1]<bbox[3]<=h):
                problems.append(f'{camera}: invalid bbox at {i}: {bbox}')
            q=r.get('quality')
            if not isinstance(q,dict) or not num(q.get('area_ratio')) or not num(q.get('blur_var')) or not isinstance(q.get('border_touch'),bool):
                problems.append(f'{camera}: quality missing/invalid at {i}')
            elif isinstance(bbox,list) and len(bbox)==4 and all(num(x) for x in bbox):
                area=(bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(w*h)
                if abs(q['area_ratio']-area)>1e-6:
                    problems.append(f'{camera}: wrong area_ratio at {i}')
                touch=(bbox[0]<=0 or bbox[1]<=0 or bbox[2]>=w or bbox[3]>=h)
                if q['border_touch']!=touch:
                    problems.append(f'{camera}: wrong border_touch at {i}')
            classes[r.get('class_name')]+=1
        errs=list(read_jsonl(d/'errors.jsonl'))
        if errs: problems.append(f'{camera}: {len(errs)} error records (investigate)')
        cors=list(read_jsonl(d/'bbox_corrections.jsonl'))
        ranked=[]; sev=Counter()
        for i,r in enumerate(cors):
            if r.get('video_id')!=vid: problems.append(f'{camera}: bbox correction video ID mismatch {i}')
            try: pct,shift,level=calc_removed(r)
            except Exception as e:
                problems.append(f'{camera}: correction {i} bad: {e}')
                continue
            sev[level]+=1
            ranked.append({**r,'removed_area_pct':round(pct,3),'max_coord_change_px':round(shift,3),'severity':level})
        ranked.sort(key=lambda x:(-x['removed_area_pct'],-x['max_coord_change_px']))
        corrections[camera]=ranked
        qa_csv=d/f'visual_review_{camera}.csv'
        qa_summary={'exists':qa_csv.is_file()}
        if qa_csv.is_file():
            with qa_csv.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            statuses=Counter(x.get('manual_status','') for x in rows)
            qa_summary.update({'review_rows':len(rows),'statuses':dict(statuses)})
            if len(rows)!=10: problems.append(f'{camera}: expected 10 visual reviews, got {len(rows)}')
            if any(x.get('manual_status') not in ('PASS','FAIL','REVIEW') for x in rows):
                problems.append(f'{camera}: invalid/empty manual_status')
        results[camera]={'video_id':vid,'frames':len(frames),'tracks':observations,'errors':len(errs),
          'bbox_corrections':len(cors),'bbox_severity':dict(sev),'classes':dict(classes),'visual_review':qa_summary,
          'output_dir':str(d)}
        counts['frames']+=len(frames);counts['tracks']+=observations;counts['errors']+=len(errs);counts['bbox_corrections']+=len(cors)
        print(f"{camera}: frames={len(frames)} tracks={observations} errors={len(errs)} bbox_clamps={len(cors)} severity={dict(sev)} visual_review={qa_summary}")
        with (out/f'bbox_ranked_{camera}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            fields=['rank','video_id','frame_index','timestamp_ms','track_uid','removed_area_pct','max_coord_change_px','severity','bbox_original','bbox_clamped','comparison_image','manual_status','reviewer_notes']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for j,r in enumerate(ranked,1):
                writer.writerow({'rank':j,'video_id':r.get('video_id'),'frame_index':r.get('frame_index'),'timestamp_ms':r.get('timestamp_ms'),
                    'track_uid':r.get('track_uid'),'removed_area_pct':r['removed_area_pct'],'max_coord_change_px':r['max_coord_change_px'],
                    'severity':r['severity'],'bbox_original':json.dumps(r['bbox_original']),'bbox_clamped':json.dumps(r['bbox_clamped']),
                    'comparison_image':str((out/f'bbox_review_{camera}'/f'top_{j:02d}_frame_{r["frame_index"]}.jpg').relative_to(root)) if j<=args.render_top else '',
                    'manual_status':'PENDING','reviewer_notes':''})
    if not args.skip_images and args.render_top:
        import cv2
        vroot=root/'data/CityFlow-NL/CityFlow_selected'
        for camera in ('c007','c034'):
            path=vroot/camera/'vdo.avi'; dest=out/f'bbox_review_{camera}';dest.mkdir(exist_ok=True)
            cap=cv2.VideoCapture(str(path))
            if not cap.isOpened(): problems.append(f'{camera}: could not open video for top QA images');continue
            for rank,r in enumerate(corrections[camera][:args.render_top],1):
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(r['frame_index']))
                ok,bgr=cap.read()
                if not ok:problems.append(f'{camera}: could not read frame {r["frame_index"]}');continue
                red=bgr.copy(); green=bgr.copy()
                for canvas,key,color in ((red,'bbox_original',(0,0,255)),(green,'bbox_clamped',(0,255,0))):
                    b=r[key];cv2.rectangle(canvas,tuple(round(x) for x in b[:2]),tuple(round(x) for x in b[2:]),color,3)
                cv2.putText(red,'ORIGINAL (RED)',(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
                cv2.putText(green,'CLAMPED (GREEN)',(20,45),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,255,0),3)
                canvas=cv2.hconcat([cv2.resize(red,(960,540)),cv2.resize(green,(960,540))])
                cv2.imwrite(str(dest/f'top_{rank:02d}_frame_{r["frame_index"]}.jpg'),canvas)
            cap.release()
    report={'schema_version':'cctv-3.0','scope':'B three-video post-export validation','results':results,
            'totals':dict(counts),'issues':problems,
            'pixel_parity':'PASS on 5 sample frames each for c007/c033/c034 (from user supplied earlier tests; not rerun by this script)',
            'cross_team_agreement':'PENDING: Lite-DETR substitution and det01 namespace with F/G',
            'bbox_review_status':'PENDING: inspect new ranked images, do not mark unreviewed cases PASS'}
    (out/'three_video_qa_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TOTAL:',dict(counts),'ISSUES:',len(problems),'REPORT:',out/'three_video_qa_report.json')
    for x in problems[:20]:print('ISSUE:',x)
    raise SystemExit(1 if problems else 0)

if __name__=='__main__':main()
