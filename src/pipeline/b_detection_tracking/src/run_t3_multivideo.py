#!/usr/bin/env python3
"""CPU ByteTrack T3 for c007+c034 Lite-DETR JSONL. Never overwrites results."""
from pathlib import Path
import argparse, json, hashlib, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_litedetr_bytetrack_cpu import read_records, tracker_args, input_boxes
from ultralytics.trackers.byte_tracker import BYTETracker


JOBS = {
    "c007": ("cityflow_S02_c007", 983),
    "c034": ("cityflow_S05_c034", 1713),
}

def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--project',type=Path,default=Path('.'))
 ap.add_argument('--videos',nargs='+',choices=list(JOBS),default=list(JOBS))
 args=ap.parse_args()
 root=args.project.resolve()
 destroot=root/'runs/b_litedetr_multivideo/t3_tracking'
 destroot.mkdir(parents=True,exist_ok=True)
 for camera in args.videos:
  vid,expected=JOBS[camera]
  source=root/f'runs/b_litedetr_multivideo/detections_{camera}_pts.jsonl'
  target=destroot/f'tracking_{camera}_T3.jsonl'
  report=destroot/f'tracking_{camera}_T3_report.json'
  if not source.is_file():raise FileNotFoundError(f'먼저 Drive의 탐지 JSONL을 WSL 프로젝트에 복사하세요: {source}')
  if target.exists() or report.exists():raise FileExistsError(f'결과 덮어쓰기 방지: {target} 또는 {report}. 기존 완료물을 유지하세요.')
  records=read_records(source)
  if len(records)!=expected or any(r['video_id']!=vid for r in records):
   raise ValueError(f'{camera}: 기대 {expected}개 프레임, video_id={vid}; 실제 {len(records)}개')
  confs=[float(d['confidence']) for r in records for d in r['detections']]
  if confs and min(confs)<.10-1e-6:raise ValueError('탐지 입력의 confidence 하한 위반')
  params=tracker_args(False)
  params.track_high_thresh=.35
  params.track_low_thresh=.10
  params.new_track_thresh=.35
  params.track_buffer=30
  params.match_thresh=.70
  params.fuse_score=False
  tracker=BYTETracker(args=params) # new tracker for EACH video
  counts=dict(video_id=vid,frames=len(records),input_detections=0,raw_track_observations=0,published_eligible_conf025=0,unique_local_track_ids=0,zero_track_frames=0)
  track_ids=set()
  partial=target.with_suffix('.jsonl.partial')
  if partial.exists():raise FileExistsError(f'이전 미완료 파일을 먼저 확인하세요: {partial}')
  try:
   with partial.open('x',encoding='utf-8') as f:
    for r in records:
     dets=[d for d in r['detections'] if float(d['confidence'])>=.10]
     counts['input_detections']+=len(dets)
     tracks=tracker.update(input_boxes(dets,int(r['height']),int(r['width'])))
     if len(tracks)==0:counts['zero_track_frames']+=1
     objs=[]; local_ids=set()
     for t in tracks:
      if len(t)<7:raise ValueError(f'Unexpected ByteTrack row: {t}')
      x1,y1,x2,y2,tid,score,cls=t[:7]
      tid=str(int(tid))
      if tid in local_ids:raise ValueError(f'Duplicate ID within frame: {vid} {r["frame_index"]} {tid}')
      local_ids.add(tid);track_ids.add(tid)
      score=float(score)
      if score>=.25:counts['published_eligible_conf025']+=1
      objs.append(dict(track_uid=f'det01:{vid}:{tid}',local_track_id=tid,bbox=[float(x1),float(y1),float(x2),float(y2)],confidence=score,class_id=int(cls)))
     counts['raw_track_observations']+=len(objs)
     f.write(json.dumps(dict(record_type='tracking_experiment',video_id=vid,frame_index=int(r['frame_index']),timestamp_ms=int(r['actual_timestamp_ms']),requested_timestamp_ms=int(r['requested_timestamp_ms']),tracks=objs),ensure_ascii=False)+'\n')
   partial.rename(target)
  except Exception:
   print(f'추적 실패. 부분 결과: {partial}',file=sys.stderr)
   raise
  counts['unique_local_track_ids']=len(track_ids)
  counts.update(input_sha256=sha256(source),output_sha256=sha256(target),parameters=dict(track_high_thresh=.35,track_low_thresh=.10,new_track_thresh=.35,match_thresh=.70,track_buffer=30,fuse_score=False,input_min_confidence=.10,published_min_confidence=.25))
  with report.open('x',encoding='utf-8') as f:json.dump(counts,f,ensure_ascii=False,indent=2)
  print(f'PASS: {vid} | frames {len(records)} | raw tracks {counts["raw_track_observations"]} | >=.25 {counts["published_eligible_conf025"]} | {target}',flush=True)
 print('PASS: T3 tracking completed, B3/B4 formal export remains',flush=True)

if __name__=='__main__':main()
