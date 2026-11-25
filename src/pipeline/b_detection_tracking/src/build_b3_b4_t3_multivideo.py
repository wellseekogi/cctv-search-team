#!/usr/bin/env python3
"""B v5.0 B3/B4: Lite-DETR T3 tracking -> candidate cctv-3.0 observations.

CPU only. Reads full video-specific detection/tracking JSONLs, decodes the original
video ONCE in frame_index order for actual-image quality metrics. Refuses to
replace any existing output. Does not invoke YOLO or Lite-DETR inference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import cv2
import numpy as np
import yaml

VIDEO_ID=None
COCO = {
    1: "person"
}
REQUIRED_COUNT=None
SOURCE_ID='det01' # v5.0 example. Do NOT merge into old YOLO det01 namespace.


def jsonl(path):
    with path.open(encoding='utf-8') as f:
        for ln,line in enumerate(f,1):
            if not line.strip(): continue
            try: val=json.loads(line)
            except Exception as e: raise ValueError(f'{path}:{ln}: invalid JSON: {e}') from e
            if not isinstance(val,dict): raise ValueError(f'{path}:{ln}: expected object')
            yield val


def sha256(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def ver(name):
    try:return metadata.version(name)
    except metadata.PackageNotFoundError:return None


def clamp_bbox(b,w,h):
    if len(b)!=4: raise ValueError('bbox requires 4 numbers')
    raw=[float(v) for v in b]
    if not all(math.isfinite(x) for x in raw): raise ValueError('bbox contains NaN/inf')
    out=[max(0.,min(float(w),raw[0])),max(0.,min(float(h),raw[1])),
         max(0.,min(float(w),raw[2])),max(0.,min(float(h),raw[3]))]
    if out[0]>=out[2] or out[1]>=out[3]: raise ValueError('invalid or degenerate bbox')
    return raw,out


def quality_metrics(rgb,bbox):
    H,W=rgb.shape[:2]
    x1,y1,x2,y2=bbox
    # v5 B4: grayscale object crop resized to 128x128 -> Laplacian variance
    a,b,c,d=math.floor(x1),math.floor(y1),math.ceil(x2),math.ceil(y2)
    crop=rgb[b:d,a:c]
    if crop.size==0: raise ValueError('empty object crop')
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    gray=cv2.resize(gray,(128,128),interpolation=cv2.INTER_LINEAR)
    blur=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    area=float((x2-x1)*(y2-y1)/(W*H))
    return {'area_ratio':area,'blur_var':blur,
            'border_touch':bool(x1<=0 or y1<=0 or x2>=W or y2>=H)}


def sample_video_path(root,explicit):
    if explicit: return explicit.expanduser().resolve()
    candidates=[
       root/'data/CityFlow-NL/CityFlow_selected/c033/vdo.avi',
       root/'data/c033/vdo.avi',
       root/'data/CityFlow_selected/c033/vdo.avi',
    ]
    return next((x for x in candidates if x.is_file()),candidates[0])


def run(args):
    root=args.project.expanduser().resolve()
    detection=(args.detection_path if args.detection_path.is_absolute() else root/args.detection_path).resolve()
    tracking=(args.tracking_path if args.tracking_path.is_absolute() else root/args.tracking_path).resolve()
    video=args.video_path.expanduser().resolve()
    destination=(args.output_dir if args.output_dir.is_absolute() else root/args.output_dir).resolve()
    for p in (detection,tracking,video):
        if not p.is_file(): raise FileNotFoundError(f'Input missing: {p}')
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite output directory: {destination}')
    global VIDEO_ID, REQUIRED_COUNT
    VIDEO_ID=args.video_id
    REQUIRED_COUNT=args.expected_frames
    destination.parent.mkdir(parents=True,exist_ok=True)
    dets=list(jsonl(detection)); tracks=list(jsonl(tracking))
    if len(dets)!=REQUIRED_COUNT or len(tracks)!=REQUIRED_COUNT:
        raise ValueError(f'Expected {REQUIRED_COUNT} frame rows; got detection={len(dets)}, tracking={len(tracks)}')
    if len(dets)!=REQUIRED_COUNT or len(tracks)!=REQUIRED_COUNT: raise ValueError('Wrong frame count')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f'Cannot decode video: {video}')
    cap_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); cap_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if cap_w<=0 or cap_h<=0: raise ValueError('Video frame dimensions invalid')

    tmp=Path(tempfile.mkdtemp(prefix='.b3_b4_staging_',dir=str(destination.parent)))
    frame_path=tmp/'frames.jsonl'; track_path=tmp/'tracks.jsonl'; err_path=tmp/'errors.jsonl'
    corr_path=tmp/'bbox_corrections.jsonl'; visual=tmp/'validation_visuals'; visual.mkdir()
    counter=0; n_obs=0; n_err=0; n_corr=0; n_blank=0; last_frame_idx=-1
    previous_track_times={}; last_req=-1
    # Choose spread-out frames for reviewer manual comparison (no automatic visual PASS claim)
    visual_indices=set(np.linspace(0,REQUIRED_COUNT-1,10,dtype=int).tolist())
    try:
        with frame_path.open('w',encoding='utf-8') as ff,track_path.open('w',encoding='utf-8') as tf,\
             err_path.open('w',encoding='utf-8') as ef,corr_path.open('w',encoding='utf-8') as cf:
            for i,(d,t) in enumerate(zip(dets,tracks)):
                req=i*200
                if d.get('video_id')!=VIDEO_ID or t.get('video_id')!=VIDEO_ID: raise ValueError(f'Video ID mismatch at row {i}')
                if int(d['requested_timestamp_ms'])!=req or int(t['requested_timestamp_ms'])!=req: raise ValueError(f'Requested PTS mismatch at {i}')
                actual=int(d['actual_timestamp_ms'])
                if actual!=int(t['timestamp_ms']) or abs(actual-req)>100: raise ValueError(f'Actual PTS mismatch at {i}')
                fi=int(d['frame_index'])
                if fi!=int(t['frame_index']) or fi<=last_frame_idx: raise ValueError(f'frame_index invalid at {i}')
                if actual<=last_req: raise ValueError(f'Actual PTS not ascending at {i}')
                last_frame_idx=fi; last_req=actual
                w,h=int(d['width']),int(d['height'])
                if w!=cap_w or h!=cap_h: raise ValueError(f'Video dimensions disagree at {i}: {w}x{h} vs {cap_w}x{cap_h}')
                # Only sequential reads; do not seek the same video on every sampled frame.
                while counter<=fi:
                    ok,bgr=cap.read()
                    if not ok: break
                    counter+=1
                if counter<=fi:
                    msg=f'Frame {fi} failed during sequential decode; requested {req}ms, actual {actual}ms'
                    ef.write(json.dumps({'schema_version':'cctv-3.0','video_id':VIDEO_ID,'frame_index':fi,
                      'requested_timestamp_ms':req,'timestamp_ms':actual,'error_code':'DECODE_FAILED',
                      'message':msg,'actual_time_error_ms':actual-req},ensure_ascii=False)+'\n')
                    n_err+=1
                    ff.write(json.dumps({'schema_version':'cctv-3.0','video_id':VIDEO_ID,'frame_index':fi,
                      'requested_timestamp_ms':req,'timestamp_ms':actual,'time_source':d.get('time_source'),
                      'source_offset_ms':d.get('source_offset_ms',0),'width':w,'height':h,
                      'decode_status':'failed','detector_status':'failed'},ensure_ascii=False)+'\n')
                    continue
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                ff.write(json.dumps({'schema_version':'cctv-3.0','video_id':VIDEO_ID,'frame_index':fi,
                  'requested_timestamp_ms':req,'timestamp_ms':actual,'time_source':d.get('time_source'),
                  'source_offset_ms':d.get('source_offset_ms',0),'width':w,'height':h,
                  'decode_status':'success','detector_status':'ran'},ensure_ascii=False)+'\n')
                entries=t.get('tracks',[])
                if not isinstance(entries,list):raise ValueError(f'Invalid tracks list at {i}')
                accepted=[]; seen=set()
                for item in entries:
                    conf=float(item['confidence'])
                    if not math.isfinite(conf): raise ValueError(f'Nonfinite track score at {i}')
                    # v5: published pred observations must be >= .25; T3 received detections >= .10 before tracking.
                    if conf<.25: continue
                    cls=int(item['class_id'])
                    if cls not in COCO: raise ValueError(f'Unexpected Lite-DETR class_id {cls} at {i}')
                    raw,box=clamp_bbox(item['bbox'],w,h)
                    local_raw=str(item['local_track_id']).removeprefix('t')
                    if not local_raw.isdecimal():raise ValueError(f'Unexpected local track ID: {local_raw}')
                    local=f't{int(local_raw):06d}'
                    uid=f'{SOURCE_ID}:{VIDEO_ID}:{local}'
                    if uid in seen:raise ValueError(f'Duplicate track ID at time {actual}: {uid}')
                    if uid in previous_track_times and actual<=previous_track_times[uid]:raise ValueError(f'Duplicate/out-of-order track {uid}')
                    previous_track_times[uid]=actual;seen.add(uid)
                    row={'schema_version':'cctv-3.0','source':'pred','track_source_id':SOURCE_ID,
                         'video_id':VIDEO_ID,'frame_index':fi,'timestamp_ms':actual,'local_track_id':local,
                         'track_uid':uid,'class_name':COCO[cls],'bbox':box,'confidence':conf,
                         'quality':quality_metrics(rgb,box)}
                    tf.write(json.dumps(row,ensure_ascii=False)+'\n');n_obs+=1
                    accepted.append(row)
                    if raw!=box:
                        cf.write(json.dumps({'video_id':VIDEO_ID,'frame_index':fi,'timestamp_ms':actual,
                         'track_uid':uid,'bbox_original':raw,'bbox_clamped':box,'reason':'out_of_frame_bounds'},ensure_ascii=False)+'\n')
                        n_corr+=1
                if not accepted:n_blank+=1
                if i in visual_indices:
                    overlay=bgr.copy()
                    for row in accepted:
                        x1,y1,x2,y2=(int(round(v)) for v in row['bbox'])
                        cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,230,255),2)
                        cv2.putText(overlay,row['local_track_id'],(x1,max(20,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,230,255),1)
                    cv2.imwrite(str(visual/f'{i:04d}_pts_{actual}.jpg'),overlay,[cv2.IMWRITE_JPEG_QUALITY,88])
        cap.release()
        # Treat incomplete video decode as blocker, never pass incomplete formal deliverable.
        if n_err: raise RuntimeError(f'{n_err} frames failed to decode; see staging {tmp}. Formal export not finalized')
        if n_obs==0: raise ValueError('No formal track observations produced')
        vision={'vision':{
            'candidate_step_ms':200,'max_frame_error_ms':100,
            'detector_model':'Lite-DETR / Lite-DINO H2L2-(2+1)x3',
            'checkpoint':'r50_s2ex3_49.9.pth',
            'tracker':'bytetrack.yaml',
            'input_size':{'short_side':480,'max_side':854},
            'confidence':.10,'publication_min_confidence':.25,'nms_iou':.5,
            'tracker_settings':{'track_high_thresh':.35,'track_low_thresh':.10,
                'new_track_thresh':.35,'track_buffer':30,'match_thresh':.70,'fuse_score':False},
            'sample_processing_rate_hz':5,
            'coordinate_system':'original_image_xyxy_pixels',
            'source_id':SOURCE_ID,
            'class_map':COCO,
            'quality':{'area_ratio':'clamped_bbox_area / original_frame_area',
                'blur_var':'Laplacian variance of grayscale bbox crop resized 128x128',
                'border_touch':'x1=0 or y1=0 or x2=width or y2=height'},
            'decoder':'OpenCV VideoCapture sequential RGB conversion for quality; PTS from saved PyAV b_video_io samples',
            'experiment':'T3',
            'versions':{'python':sys.version.split()[0], **{name:ver(name) for name in ('opencv-python','opencv-python-headless','numpy','PyYAML','ultralytics','torch')}},
            'baseline_note':'YOLO11n is historical benchmark only; final B prediction uses Lite-DETR',
            'model_change_approval':'PENDING F/G agreement on v5 YOLO11n->Lite-DETR substitution',
            'source_id_collision_warning':'det01 is v5 example; never mix these UIDs with old YOLO det01 run'},}
        (tmp/'vision.yaml').write_text(yaml.safe_dump(vision,allow_unicode=True,sort_keys=False),encoding='utf-8')
        report={'schema_version':'cctv-3.0','video_id':VIDEO_ID,'sampled_frames':len(dets),
          'tracks_observations':n_obs,'frames_no_tracks':n_blank,'errors':n_err,'bbox_clamps':n_corr,
          'visual_samples_saved':len(visual_indices),'visual_manual_review':'PENDING',
          'experiment':'T3','tracking_input_confidence':0.10,'publication_min_confidence':0.25,
          'tracker_settings':{'track_high_thresh':0.35,'track_low_thresh':0.10,
               'new_track_thresh':0.35,'match_thresh':0.70,'track_buffer':30,'fuse_score':False},
          'input_sha256':{detection.name:sha256(detection),
              tracking.name:sha256(tracking),'video_file':sha256(video)},
          'limits':['Per-video candidate output; the three-video integration and visual QA need independent checks',
                    'Only final tracked objects are exported; untracked detections do not become observations',
                    'Visual inspection of 10 rendered boxes still required',
                    'T3 selected using c033 development footage; c007/c034 not used for tuning',
                    'Do not merge det01 UIDs with old YOLO det01 UIDs',
                    'Selected frame pixels are read by OpenCV index; cross-decoder pixel equivalence must be checked against read_frame'],
          'files_sha256':{'frames.jsonl':sha256(frame_path),'tracks.jsonl':sha256(track_path),
               'errors.jsonl':sha256(err_path),'bbox_corrections.jsonl':sha256(corr_path)}}
        (tmp/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (tmp/'README.md').write_text('B v5.0 Lite-DETR T3 candidate export. Read validation_report.json limits before sharing.\n'
            'Run: python src/build_b3_b4_t3_multivideo.py with video-specific arguments\n'
            'WARNING: SOURCE_ID det01 follows v5 illustration but COLLIDES with old YOLO if merged.\n'
            'Visual QA and 3-video validation remain pending. Do not label this final handoff.\n',encoding='utf-8')
        tmp.rename(destination)
        print('PASS: B3/B4 export generated',destination)
        print(f'frames={len(dets)}, tracks={n_obs}, errors={n_err}, bbox_clamps={n_corr}, visuals={len(visual_indices)}')
        print('PENDING: visual manual review, read_frame pixel parity, other videos, F/G model/source namespace agreement')
    except Exception:
        cap.release()
        print('STAGING PRESERVED FOR DEBUG:',tmp,file=sys.stderr)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('.'))
    p.add_argument('--video-id',type=str,required=True)
    p.add_argument('--expected-frames',type=int,required=True)
    p.add_argument('--detection-path',type=Path,required=True)
    p.add_argument('--video-path',type=Path,required=True)
    p.add_argument('--tracking-path',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__': main()
