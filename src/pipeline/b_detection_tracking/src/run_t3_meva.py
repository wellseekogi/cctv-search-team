#!/usr/bin/env python3
"""Run ByteTrack T3 on one MEVA Lite-DETR detection JSONL. No overwrites."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_litedetr_bytetrack_cpu import read_records, tracker_args, input_boxes
from ultralytics.trackers.byte_tracker import BYTETracker


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video-id', required=True)
    ap.add_argument('--detection-path', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--expected-frames', type=int, default=None)
    args = ap.parse_args()

    source = args.detection_path.resolve()
    assert source.is_file(), f'Missing detection JSONL: {source}'
    dest = args.output_dir.resolve()
    assert not dest.exists(), f'Refusing to overwrite existing directory: {dest}'
    records = read_records(source)
    assert records, 'Empty detection JSONL'
    if args.expected_frames is not None:
        assert len(records) == args.expected_frames, (len(records), args.expected_frames)
    assert all(r['video_id'] == args.video_id for r in records), 'video_id mismatch'
    assert all(int(r['requested_timestamp_ms']) == i * 200 for i, r in enumerate(records)), 'Expected 200ms samples'
    assert all(float(d['confidence']) >= .10 - 1e-6 for r in records for d in r['detections']), 'Input confidence below .10'
    all_classes = {int(d['class_id']) for r in records for d in r['detections']}
    assert all_classes.issubset({1}), f'Expected person COCO category ID=1; got {all_classes}. Check detector class mapping!'

    params = tracker_args(False)
    params.track_high_thresh = .35
    params.track_low_thresh = .10
    params.new_track_thresh = .35
    params.track_buffer = 30
    params.match_thresh = .70
    params.fuse_score = False
    tracker = BYTETracker(args=params)  # fresh tracker per video

    dest.mkdir(parents=True)
    target = dest / 'tracking_T3.jsonl'
    report = dest / 'tracking_T3_report.json'
    partial = dest / 'tracking_T3.jsonl.partial'
    unique_ids = set()
    n_raw, n_eligible, zero_frames, n_input = 0, 0, 0, 0
    try:
        with partial.open('x', encoding='utf-8') as f:
            for r in records:
                detections = [d for d in r['detections'] if float(d['confidence']) >= .10]
                n_input += len(detections)
                tracks = tracker.update(input_boxes(detections, int(r['height']), int(r['width'])))
                if len(tracks) == 0:
                    zero_frames += 1
                objs, local_ids = [], set()
                for t in tracks:
                    assert len(t) >= 7, f'Unexpected ByteTrack row: {t}'
                    x1, y1, x2, y2, tid, score, cls = t[:7]
                    tid = str(int(tid))
                    assert tid not in local_ids, f'Duplicate ID in frame: {tid}'
                    assert int(cls) == 1, f'Non-person class in tracker output: {cls}'
                    local_ids.add(tid)
                    unique_ids.add(tid)
                    score = float(score)
                    if score >= .25:
                        n_eligible += 1
                    objs.append({
                        'track_uid': f'det01:{args.video_id}:{tid}',
                        'local_track_id': tid,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': score,
                        'class_id': int(cls),
                    })
                n_raw += len(objs)
                row = {
                    'record_type': 'tracking_experiment',
                    'video_id': args.video_id,
                    'frame_index': int(r['frame_index']),
                    'timestamp_ms': int(r['actual_timestamp_ms']),
                    'requested_timestamp_ms': int(r['requested_timestamp_ms']),
                    'tracks': objs,
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        partial.rename(target)
        payload = {
            'video_id': args.video_id,
            'frames': len(records),
            'input_detections': n_input,
            'raw_track_observations': n_raw,
            'published_eligible_conf025': n_eligible,
            'unique_local_track_ids': len(unique_ids),
            'zero_track_frames': zero_frames,
            'input_sha256': sha256(source),
            'output_sha256': sha256(target),
            'parameters': {
                'track_high_thresh': .35, 'track_low_thresh': .10,
                'new_track_thresh': .35, 'match_thresh': .70,
                'track_buffer': 30, 'fuse_score': False,
                'input_min_confidence': .10,
                'published_min_confidence': .25,
            },
        }
        with report.open('x', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        print(f'FAILED: partial results at {partial}', file=sys.stderr)
        raise
    print(f'PASS: {args.video_id} | frames={len(records)} | raw_tracks={n_raw} | >=.25={n_eligible}')
    print('T3 tracking complete; B3/B4 export remains')


if __name__ == '__main__':
    main()
