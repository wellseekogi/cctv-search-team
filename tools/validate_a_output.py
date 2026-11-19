import hashlib
import json
from collections import Counter
from integrations.c_sample_contract import validate_sample, crop_box

def sha(data):
    return hashlib.sha256(data).hexdigest()

def rows(data):
    return [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]

def c_checks():
    return {"validate_sample":validate_sample,"crop_box":crop_box}

def verify_output(events, samples, info, frames, tracks, manifest, config, checks):
    frame_map = {f['frame_index']: f for f in frames}
    track_map = {(t['frame_index'], t['track_uid']): t for t in tracks}
    by_id = {s['item_id']: s for s in samples}
    assert len(by_id) == len(samples)
    evs = {e['event_id']: e for e in events}
    assert len(evs) == len(events)
    groups = {}
    scene_minutes = Counter()
    crops = []
    for s in samples:
        checks['validate_sample'](s, manifest)
        f = frame_map[s['frame_index']]
        assert s['video_id'] == f['video_id'] and s['timestamp_ms'] == f['timestamp_ms']
        assert all(e in evs for e in s['event_ids'])
        key = s['track_uid'] if s['representation'] == 'object' else '__scene__'
        groups.setdefault(key, []).append(s['timestamp_ms'])
        if s['representation'] == 'object':
            t = track_map[s['frame_index'], s['track_uid']]
            assert t['class_name'] == 'person' and s['bbox'] == t['bbox']
            assert s['track_source_id'] == t['track_source_id']
            crops.append(dict(item_id=s['item_id'], expected_crop_bbox=checks['crop_box'](
                s['bbox'], f['width'], f['height']), **{k:s[k] for k in ('video_id','frame_index','timestamp_ms','bbox','track_uid')}))
        else:
            scene_minutes[s['timestamp_ms'] // 60000] += 1
    for key, times in groups.items():
        times.sort()
        gap = config['sampling']['min_scene_gap_ms' if key == '__scene__' else 'min_object_gap_ms']
        assert all(b-a >= gap for a,b in zip(times,times[1:]))
        if key != '__scene__':
            assert len(times) <= config['sampling']['max_object_per_track']
    assert all(n <= config['sampling']['max_scene_per_minute'] for n in scene_minutes.values())
    for ev in events:
        assert ev['event_type'] != 'VIDEO_END'
        for rep in ('object','scene'):
            item = ev[rep+'_item_id']
            assert item is None or ev['event_id'] in by_id[item]['event_ids']
    assert info['counts'] == dict(events=len(events), object=sum(s['representation']=='object' for s in samples),
                                 scene=sum(s['representation']=='scene' for s in samples))
    return crops
