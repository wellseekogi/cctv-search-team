"""Fixed schedule on the original time axis; select actual candidates within 100ms."""
from collections import Counter, defaultdict

from .core import stable_id


def select_uniform(config, videos, frames, tracks, exclusions, run_id, selected_frames=None):
    settings = config["sampling"]
    step = settings["fixed_step_ms"]
    preview = config.get("execution_mode") == "annotation_preview"
    partial = config.get("execution_mode") == "observed_partial"
    reason = "motion" if selected_frames is not None else "uniform"
    config_hash = stable_id("cfg", config)
    sources = sorted({o["track_source_id"] for o in tracks})
    source = sources[0] if sources else config.get("track_source_id")
    by_frame = defaultdict(list)
    for row in tracks:
        by_frame[row["video_id"], row["frame_index"]].append(row)
    samples, decisions = [], []
    selected = {}
    object_times, scene_times = defaultdict(list), defaultdict(list)
    scene_counts = Counter()

    def choose(frame, requested, observation=None):
        video, t = frame["video_id"], frame["timestamp_ms"]
        representation = "object" if observation else "scene"
        uid = observation["track_uid"] if observation else None
        key = video, frame["frame_index"], representation, uid
        times = object_times[uid] if observation else scene_times[video]
        gap = settings["min_object_gap_ms"] if observation else settings["min_scene_gap_ms"]
        cap = settings["max_object_per_track"] if observation else settings["max_scene_per_minute"]
        count = len(times) if observation else scene_counts[video, t // 60000]
        item = None
        if any(x["video_id"] == video and x["start_ms"] <= t < x["end_ms"] for x in exclusions):
            status = "QUERY_WINDOW"
        elif key in selected:
            status, item = "MERGED", selected[key]
        elif any(abs(t-other) < gap for other in times):
            status = "INTERVAL"
        elif count >= cap:
            status = "CAP"
        else:
            status = "ACCEPTED"
            item = dict(schema_version="cctv-3.0", item_id=stable_id("item", source, config_hash, video,
                        frame["frame_index"], representation, uid), video_id=video,
                        frame_index=frame["frame_index"], timestamp_ms=t, representation=representation,
                        track_uid=uid, bbox=observation["bbox"] if observation else None,
                        event_ids=[], reasons=[reason], policy=settings["policy"], source_run_id=run_id,
                        track_source_id=source)
            if preview:
                item.update(preview_only=True, timestamp_verified=False, query_exclusion_status="not_provided")
            if partial:
                item.update(provisional=True, timestamp_source="B_frames_pts", media_verified=False,
                            query_exclusion_status=config.get("query_exclusion_status", "not_provided"))
            selected[key] = item
            samples.append(item)
            times.append(t)
            if not observation:
                scene_counts[video, t // 60000] += 1
        decisions.append(dict(video_id=video, requested_timestamp_ms=requested, timestamp_ms=t,
                         frame_index=frame["frame_index"], representation=representation, track_uid=uid,
                         status=status, item_id=item["item_id"] if item else None, min_gap_ms=gap, cap=cap))

    for video, m in sorted(videos.items()):
        candidates = [f for f in frames if f["video_id"] == video]
        if selected_frames is not None:
            for frame in sorted((f for f in selected_frames if f["video_id"] == video), key=lambda f:f["timestamp_ms"]):
                choose(frame, frame["timestamp_ms"])
                for o in sorted(by_frame[video, frame["frame_index"]], key=lambda x:x["track_uid"]):
                    choose(frame, frame["timestamp_ms"], o)
            continue
        # The schedule is 0, step, 2*step ... even for a cropped source interval.
        first = ((m["available_start_ms"] + step - 1) // step) * step
        for requested in range(first, m["available_end_ms"], step):
            frame = min(candidates, key=lambda f: (abs(f["timestamp_ms"]-requested), f["timestamp_ms"], f["frame_index"]), default=None)
            if frame is None or abs(frame["timestamp_ms"]-requested) > 100:
                decisions.append(dict(video_id=video, requested_timestamp_ms=requested, status="NO_CANDIDATE"))
                continue
            if frame["decode_status"] == "failed":
                decisions.append(dict(video_id=video, requested_timestamp_ms=requested, status="DECODE_FAILED"))
                continue
            choose(frame, requested)
            for o in sorted(by_frame[video, frame["frame_index"]], key=lambda x: x["track_uid"]):
                choose(frame, requested, o)
    samples.sort(key=lambda s: (s["video_id"], s["timestamp_ms"], s["representation"], s["track_uid"] or ""))
    counts = Counter(s["representation"] for s in samples)
    summary = dict(schema_version="cctv-3.0", status="success", source_run_id=run_id, policy=settings["policy"],
                   config_hash=config_hash, fixed_step_ms=step,
                   counts={"events": 0, "object": counts["object"], "scene": counts["scene"]},
                   event_types={}, decision_statuses=dict(sorted(Counter(d["status"] for d in decisions).items())),
                   selection_decisions=decisions)
    if preview:
        summary.update(execution_mode="annotation_preview", final_evaluation_ready=False,
                       candidate_coverage="only supplied annotation frames; no frames inferred")
    if partial:
        summary.update(execution_mode="observed_partial", final_evaluation_ready=False,
                       completion_status="pending", query_exclusion_status=config.get("query_exclusion_status", "not_provided"))
    return [], samples, summary
