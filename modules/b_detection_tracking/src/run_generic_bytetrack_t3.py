#!/usr/bin/env python3
"""Generic Lite-DETR detection JSONL -> ByteTrack T3 tracking JSONL.

Colab: %pip install -q ultralytics
python run_generic_bytetrack_t3.py --input /path/detections_VIDEO_pts.jsonl --output-dir /path/track_t3

Tracking JSONL is an intermediate product, NOT the finalized B3/B4 cctv-3.0 schema.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker


def to_boxes(detections, height, width):
    rows = []
    for d in detections:
        if int(d.get('class_id', -1)) != 1:
            continue
        b = [float(x) for x in d['bbox']]
        score = float(d['confidence'])
        if not (all(math.isfinite(x) for x in b) and math.isfinite(score)):
            continue
        x1, y1, x2, y2 = b
        x1, x2 = max(0., min(float(width), x1)), max(0., min(float(width), x2))
        y1, y2 = max(0., min(float(height), y1)), max(0., min(float(height), y2))
        if x2 <= x1 or y2 <= y1 or not (0.10 <= score <= 1.0):
            continue
        rows.append([x1, y1, x2, y2, score, 1.])
    arr = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    return Boxes(torch.from_numpy(arr), orig_shape=(height, width)).cpu().numpy()


def run(input_path, output_dir, expected_frames=None):
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'tracking_T3_{input_path.stem.removeprefix("detections_")}.jsonl'
    summary_file = output_dir / f'summary_T3_{input_path.stem.removeprefix("detections_")}.json'
    if output_file.exists() or summary_file.exists():
        raise FileExistsError(f'이미 결과가 존재합니다. 새 출력 폴더를 지정하세요: {output_dir}')

    # Original T3 tuning. One fresh tracker per input video.
    args = SimpleNamespace(
        tracker_type='bytetrack', track_high_thresh=0.35,
        track_low_thresh=0.10, new_track_thresh=0.35,
        track_buffer=30, match_thresh=0.70, fuse_score=False,
    )
    # Previous T3 implementation used ByteTrack's default frame_rate=30.
    tracker = BYTETracker(args=args)
    video_id = None
    seen = set()
    counts = Counter()
    last_requested = -200
    last_actual = -1
    frame_count = 0
    total_in = 0
    total_out = 0
    unique_ids = set()
    zero_tracks = 0

    try:
        with input_path.open('r', encoding='utf-8') as fin, output_file.open('x', encoding='utf-8') as fout:
            for line_number, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                vid = rec['video_id']
                if video_id is None:
                    video_id = vid
                if vid != video_id:
                    raise ValueError(f'여러 영상 혼합: line {line_number}: {vid}')
                request_ms = int(rec['requested_timestamp_ms'])
                actual_ms = int(rec['actual_timestamp_ms'])
                if request_ms != last_requested + 200:
                    raise ValueError(f'200ms 연속성이 깨짐: line {line_number}, {request_ms}')
                if abs(actual_ms - request_ms) > 100:
                    raise ValueError(f'PTS 허용 오차 초과: line {line_number}')
                if actual_ms < last_actual:
                    raise ValueError(f'실제 PTS 역순: line {line_number}')
                last_requested, last_actual = request_ms, actual_ms
                width, height = int(rec['width']), int(rec['height'])
                if width <= 0 or height <= 0:
                    raise ValueError(f'프레임 크기 오류: line {line_number}')
                dets = rec.get('detections', [])
                boxes = to_boxes(dets, height, width)
                total_in += len(boxes)
                tracks = tracker.update(boxes)
                objects = []
                for t in tracks:
                    if len(t) < 7:
                        raise ValueError(f'ByteTrack 출력 형식 오류: {t!r}')
                    x1, y1, x2, y2, tid, score, cls = map(float, t[:7])
                    if int(cls) != 1 or score < 0.25:
                        continue  # Original T3 publish threshold.
                    uid = f'det01:{vid}:{int(tid)}'
                    unique_ids.add(uid)
                    objects.append({
                        'track_uid': uid,
                        'local_track_id': str(int(tid)),
                        'bbox': [x1, y1, x2, y2],
                        'confidence': score,
                        'class_id': 1,
                        'class_name': 'person',
                    })
                total_out += len(objects)
                if not objects:
                    zero_tracks += 1
                frame_count += 1
                fout.write(json.dumps({
                    'record_type': 'tracking_experiment',
                    'video_id': vid,
                    'frame_index': int(rec['frame_index']),
                    'requested_timestamp_ms': request_ms,
                    'actual_timestamp_ms': actual_ms,
                    'timestamp_ms': actual_ms,
                    'width': width,
                    'height': height,
                    'time_source': rec.get('time_source'),
                    'source_offset_ms': rec.get('source_offset_ms', 0),
                    'tracks': objects,
                }, ensure_ascii=False) + '\n')
                if frame_count % 250 == 0:
                    print(f'T3 tracking: {frame_count} frames / {total_out} track observations', flush=True)
        if not frame_count or (expected_frames is not None and frame_count != expected_frames):
            raise ValueError(f'프레임 수 오류: {frame_count} (expected={expected_frames})')
        summary = {
            'video_id': video_id, 'frames': frame_count,
            'input_detections_used': total_in, 'track_observations': total_out,
            'unique_track_ids': len(unique_ids), 'frames_without_tracks': zero_tracks,
            'settings': vars(args), 'publish_confidence': 0.25,
            'tracking_jsonl': str(output_file),
            'note': '중간 추적 결과이며 B3/B4 최종 출력 아님',
        }
        summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception:
        output_file.unlink(missing_ok=True)
        raise
    print('PASS:', json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description='Lite-DETR JSONL -> ByteTrack T3 tracking JSONL')
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--expected-frames', type=int, default=None)
    args = parser.parse_args()
    run(args.input, args.output_dir, args.expected_frames)


if __name__ == '__main__':
    main()
