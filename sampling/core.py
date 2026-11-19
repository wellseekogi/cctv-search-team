"""Deterministic object-event sampling; no decoding or model dependency."""

import hashlib
import json
import math
from collections import Counter, defaultdict


EVENT_ORDER = {name: i for i, name in enumerate((
    "TRACK_START", "TRACK_RESUME", "POSITION_SHIFT", "QUALITY_GAIN",
    "TRACK_GAP", "VIDEO_END",
))}


class InputError(ValueError):
    """Invalid input, carrying source filename and line where available."""


def stable_id(prefix, *parts):
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False)
    return prefix + "_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def require(ok, message):
    if not ok:
        raise InputError(message)


def is_int(value):
    return type(value) is int


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def quality(observation, config):
    q = observation.get("quality")
    if not q or any(q.get(k) is None for k in ("area_ratio", "blur_var", "border_touch")):
        return None
    score = (config["area_weight"] * min(q["area_ratio"] / config["area_scale"], 1)
             + config["sharp_weight"] * q["blur_var"] / (q["blur_var"] + config["blur_scale"]))
    return score * config["border_multiplier"] if q["border_touch"] else score


def center(observation):
    x1, y1, x2, y2 = observation["bbox"]
    return [(x1 + x2) / 2, (y1 + y2) / 2]


def validate(config, manifests, frames, tracks, statuses, exclusions):
    preview = config.get("execution_mode") == "annotation_preview"
    partial = config.get("execution_mode") == "observed_partial"
    require(config.get("execution_mode", "complete") in ("complete", "annotation_preview", "observed_partial"), "unknown execution_mode")
    require(config.get("schema_version") == "cctv-3.0", "config: schema_version must be cctv-3.0")
    require(config["sampling"]["policy"] in ("object_event", "uniform", "hybrid", "motion"),
            "config: unsupported sampling policy")
    if config["sampling"]["policy"] == "motion":
        require(number(config["sampling"].get("motion_threshold")) and 0 <= config["sampling"]["motion_threshold"] <= 1,
                "config: motion_threshold must be in [0,1]")
    if config["sampling"]["policy"] == "hybrid":
        require(is_int(config["sampling"].get("periodic_scene_ms")) and config["sampling"]["periodic_scene_ms"] > 0,
                "config: periodic_scene_ms must be a positive integer")
    if config["sampling"]["policy"] == "uniform":
        require(is_int(config["sampling"].get("fixed_step_ms")) and config["sampling"]["fixed_step_ms"] > 0,
                "config: fixed_step_ms must be a positive integer")
    for k in ("min_scene_gap_ms", "min_object_gap_ms", "max_scene_per_minute", "max_object_per_track"):
        require(is_int(config["sampling"][k]) and config["sampling"][k] > 0, f"config: invalid {k}")
    e = config["event"]
    require(is_int(e["gap_after_ms"]) and e["gap_after_ms"] > 0, "config: invalid gap_after_ms")
    for k in ("position_ratio", "quality_gain", "float_epsilon", "area_scale", "blur_scale",
              "area_weight", "sharp_weight", "border_multiplier"):
        require(number(e[k]) and e[k] >= 0, f"config: invalid {k}")
    require(e["area_scale"] > 0 and e["blur_scale"] > 0, "config: quality scales must be positive")
    require(abs(e["area_weight"] + e["sharp_weight"] - 1) < 1e-9
            and e["border_multiplier"] <= 1, "config: invalid quality weights")
    videos = {}
    for line, m in enumerate(manifests, 1):
        loc = f"manifest.jsonl:{line}"
        require(m.get("schema_version") == "cctv-3.0", f"{loc}: invalid schema_version")
        v = m.get("video_id")
        require(isinstance(v, str) and v and v not in videos, f"{loc}: duplicate or invalid video_id")
        for k in ("width", "height", "duration_ms"):
            require(is_int(m.get(k)) and m[k] > 0, f"{loc}: invalid {k}")
        a, b = m.get("available_start_ms"), m.get("available_end_ms")
        require(is_int(a) and is_int(b) and 0 <= a < b <= m["duration_ms"], f"{loc}: invalid available interval")
        videos[v] = m
    frame_map, times = {}, set()
    for line, f in enumerate(frames, 1):
        loc = f"frames.jsonl:{line}"
        v, idx, t = f.get("video_id"), f.get("frame_index"), f.get("timestamp_ms")
        require(f.get("schema_version") == "cctv-3.0", f"{loc}: invalid schema_version")
        require(v in videos, f"{loc}: unknown video")
        m = videos[v]
        require(is_int(idx) and idx >= 0 and is_int(t) and m["available_start_ms"] <= t < m["available_end_ms"], f"{loc}: invalid frame/time")
        require((v, idx) not in frame_map and (v, t) not in times, f"{loc}: duplicate frame/time")
        require(f.get("decode_status") in (("unverified",) if preview else ("success", "failed")), f"{loc}: invalid decode_status")
        require(f.get("detector_status") in ("ran", "skipped", "failed"), f"{loc}: invalid detector_status")
        require(f.get("width") == m["width"] and f.get("height") == m["height"], f"{loc}: dimensions mismatch")
        require(is_int(f.get("requested_timestamp_ms")) and abs(t - f["requested_timestamp_ms"]) <= 100,
                f"{loc}: requested/actual frame error exceeds 100ms")
        frame_map[v, idx] = f
        times.add((v, t))
    seen = set()
    observation_sources = set()
    for line, o in enumerate(tracks, 1):
        loc = f"tracks.jsonl:{line}"
        v, idx, t = o.get("video_id"), o.get("frame_index"), o.get("timestamp_ms")
        require(o.get("schema_version") == "cctv-3.0", f"{loc}: invalid schema_version")
        require(v in videos and is_int(idx) and (v, idx) in frame_map, f"{loc}: unknown frame")
        f, m = frame_map[v, idx], videos[v]
        require(is_int(t) and t == f["timestamp_ms"], f"{loc}: actual timestamp mismatch")
        require(f["decode_status"] == ("unverified" if preview else "success"), f"{loc}: observation on failed frame")
        require(not preview or o.get("source") == "gt", f"{loc}: preview accepts GT only")
        require(o.get("class_name") in config["supported_classes"], f"{loc}: unsupported class")
        for key in ("track_source_id", "local_track_id"):
            require(isinstance(o.get(key), str) and o[key], f"{loc}: invalid {key}")
        observation_sources.add(o["track_source_id"])
        require(o.get("track_uid") == f'{o["track_source_id"]}:{v}:{o["local_track_id"]}', f"{loc}: track_uid mismatch")
        key = (v, o["track_uid"], t)
        require(key not in seen, f"{loc}: duplicate observation")
        seen.add(key)
        b = o.get("bbox")
        require(isinstance(b, list) and len(b) == 4 and all(number(x) for x in b), f"{loc}: invalid bbox")
        require(0 <= b[0] < b[2] <= m["width"] and 0 <= b[1] < b[3] <= m["height"], f"{loc}: bbox out of bounds")
        require(o.get("source") in ("gt", "pred"), f"{loc}: invalid source")
        require((o["source"] == "gt" and o.get("confidence") is None)
                or (o["source"] == "pred" and number(o.get("confidence")) and .25 <= o["confidence"] <= 1),
                f"{loc}: invalid confidence")
        require(o["source"] != "pred" or f["detector_status"] == "ran", f"{loc}: detector did not run")
        q = o.get("quality")
        require(q is None or isinstance(q, dict), f"{loc}: invalid quality")
        if q:
            for name in ("area_ratio", "blur_var"):
                require(q.get(name) is None or (number(q[name]) and q[name] >= 0), f"{loc}: invalid {name}")
            require(q.get("border_touch") is None or type(q["border_touch"]) is bool, f"{loc}: invalid border_touch")
            require(q.get("area_ratio") is None or q["area_ratio"] <= 1, f"{loc}: area_ratio exceeds 1")
    require(len(observation_sources) <= 1,
            "tracks.jsonl: use one track_source_id per run; run GT/pred and restarted sources separately")
    state_map = {}
    for line, s in enumerate(statuses, 1):
        loc = f"video_status.jsonl:{line}"
        v = s.get("video_id")
        require(v in videos and v not in state_map, f"{loc}: unknown/duplicate video")
        require(s.get("video_status") == ("unverified" if preview else "completion_pending" if partial else "completed"), f"{loc}: video not completed; reprocess before finalizing")
        require(s.get("available_end_ms") == videos[v]["available_end_ms"], f"{loc}: end mismatch")
        state_map[v] = s
    require(set(state_map) == set(videos), "video_status.jsonl: each video needs explicit completion")
    for line, x in enumerate(exclusions, 1):
        loc = f"query_exclusions.jsonl:{line}"
        v, a, b = x.get("video_id"), x.get("start_ms"), x.get("end_ms")
        require(v in videos and is_int(a) and is_int(b) and 0 <= a < b <= videos[v]["duration_ms"], f"{loc}: invalid exclusion interval")
    return videos


def run_sampling(config, manifests, frames, tracks, statuses, exclusions, run_id="event-trace", frame_loader=None):
    # Validate all incoming observations before filtering, so bad vehicle rows
    # cannot silently hide schema/time errors in a person-only run.
    targets = config.get("target_classes")
    if targets is None:
        return _run_sampling(config, manifests, frames, tracks, statuses, exclusions, run_id, frame_loader)
    require(isinstance(targets, list) and bool(targets)
            and all(isinstance(c, str) and c in config["supported_classes"] for c in targets)
            and len(set(targets)) == len(targets), "config: invalid target_classes")
    validate(config, manifests, frames, tracks, statuses, exclusions)
    selected = [o for o in tracks if o["class_name"] in targets]
    events, samples, summary = _run_sampling(
        config, manifests, frames, selected, statuses, exclusions, run_id, frame_loader)
    summary.update(target_classes=targets,
                   input_observations=len(tracks), target_observations=len(selected),
                   ignored_observations_by_class=dict(sorted(Counter(
                       o["class_name"] for o in tracks if o["class_name"] not in targets).items())),
                   event_semantics="sampling_trigger_not_action_label",
                   action_recognition_implemented=False)
    return events, samples, summary


def _run_sampling(config, manifests, frames, tracks, statuses, exclusions, run_id, frame_loader):
    videos = validate(config, manifests, frames, tracks, statuses, exclusions)
    if config["sampling"]["policy"] == "motion":
        from .motion import select_motion
        require(callable(frame_loader), "motion requires a frame_loader; GT boxes alone cannot provide motion scores")
        return select_motion(config, videos, frames, tracks, exclusions, run_id, frame_loader)
    if config["sampling"]["policy"] == "uniform":
        from .uniform import select_uniform
        return select_uniform(config, videos, frames, tracks, exclusions, run_id)
    preview = config.get("execution_mode") == "annotation_preview"
    partial = config.get("execution_mode") == "observed_partial"
    econf, sconf = config["event"], config["sampling"]
    config_hash = stable_id("cfg", config)
    sources = sorted({o["track_source_id"] for o in tracks})
    events, objects, scenes = [], {}, {}
    states, chosen_times = {}, defaultdict(list)
    observations = defaultdict(list)
    for o in tracks:
        observations[o["video_id"], o["frame_index"]].append(o)

    def excluded(video, timestamp):
        return any(x["video_id"] == video and x["start_ms"] <= timestamp < x["end_ms"] for x in exclusions)

    def sample(o, representation):
        track = o["track_uid"] if representation == "object" else None
        source = o["track_source_id"] if track else (sources[0] if sources else config.get("track_source_id"))
        return dict(schema_version="cctv-3.0",
                    item_id=stable_id("item", source, config_hash, o["video_id"], o["frame_index"], representation, track),
                    video_id=o["video_id"], frame_index=o["frame_index"], timestamp_ms=o["timestamp_ms"],
                    representation=representation, track_uid=track, bbox=o["bbox"] if track else None,
                    event_ids=[], reasons=[], policy=sconf["policy"], source_run_id=run_id,
                    track_source_id=source)

    def emit(o, state, kind, now, trigger):
        event = dict(schema_version="cctv-3.0",
                     event_id=stable_id("event", o["track_source_id"], config_hash, o["video_id"],
                                        o["track_uid"], state["segment_id"], kind, o["timestamp_ms"]),
                     source_run_id=run_id, track_source_id=o["track_source_id"],
                     video_id=o["video_id"], track_uid=o["track_uid"], segment_id=state["segment_id"],
                     event_type=kind, event_timestamp_ms=o["timestamp_ms"], detected_at_ms=now,
                     frame_index=o["frame_index"], bbox=o["bbox"], trigger=trigger,
                     quality_status="missing" if quality(o, econf) is None else "available",
                     object_status=None, object_item_id=None, scene_status=None, scene_item_id=None)
        events.append(event)
        return event

    def link(item, group):
        item["event_ids"] = sorted(set(item["event_ids"]) | {ev["event_id"] for ev in group})
        item["reasons"] = sorted(set(item["reasons"]) | {ev["event_type"] for ev in group}, key=lambda reason: EVENT_ORDER.get(reason, 6))

    def select_object(o, state, group):
        if not group:
            return
        key = o["video_id"], o["frame_index"], o["track_uid"]
        times = chosen_times[o["track_uid"]]
        item = None
        if excluded(o["video_id"], o["timestamp_ms"]):
            status = "QUERY_WINDOW"
        elif key in objects:
            status, item = "MERGED", objects[key]
        elif any(abs(o["timestamp_ms"] - t) < sconf["min_object_gap_ms"] for t in times):
            status = "INTERVAL"
        elif len(times) >= sconf["max_object_per_track"]:
            status = "CAP"
        else:
            status, item = "ACCEPTED", sample(o, "object")
            objects[key] = item
            times.append(o["timestamp_ms"])
            state["anchor"] = o
            state["anchor_source"] = "selected"
        if item:
            link(item, group)
        for ev in group:
            ev["object_status"] = status
            ev["object_item_id"] = item["item_id"] if item else None
            ev["object_limits"] = {"min_gap_ms": sconf["min_object_gap_ms"], "max_per_track": sconf["max_object_per_track"]}

    for video in sorted(videos):
        m = videos[video]
        vframes = sorted((f for f in frames if f["video_id"] == video), key=lambda f: f["timestamp_ms"])
        for f in vframes:
            now = f["timestamp_ms"]
            # GAP must be evaluated before observations, including a return exactly at 1000 ms.
            for uid in ([] if preview else sorted(states)):
                st = states[uid]
                last = st["last"]
                if last["video_id"] != video or st["status"] != "ACTIVE" or now - last["timestamp_ms"] < econf["gap_after_ms"]:
                    continue
                logs = [row for row in vframes if last["timestamp_ms"] < row["timestamp_ms"] <= now]
                decode = any(row["decode_status"] == "failed" for row in logs)
                other = any(row["decode_status"] != "failed" for row in logs)
                reason = "mixed" if decode and other else "decode_failure" if decode else "no_detection"
                ev = emit(last, st, "TRACK_GAP", now, {"gap_ms": now-last["timestamp_ms"],
                          "threshold_ms": econf["gap_after_ms"], "gap_reason": reason})
                select_object(last, st, [ev])
                st["status"] = "LOST"
            for o in sorted(observations[video, f["frame_index"]], key=lambda x: x["track_uid"]):
                uid = o["track_uid"]
                st = states.get(uid)
                group = []
                if st is None or st["status"] == "LOST":
                    kind = "TRACK_START" if st is None else "TRACK_RESUME"
                    segment = 0 if st is None else st["segment_id"] + 1
                    st = dict(status="ACTIVE", segment_id=segment, anchor=o, anchor_source="initial", last=o)
                    states[uid] = st
                    group.append(emit(o, st, kind, now, {"anchor_source": "initial"}))
                else:
                    anchor = st["anchor"]
                    ac, oc = center(anchor), center(o)
                    distance = math.dist(ac, oc) / math.hypot(m["width"], m["height"])
                    base = dict(anchor_ms=anchor["timestamp_ms"], anchor_source=st["anchor_source"])
                    if distance + econf["float_epsilon"] >= econf["position_ratio"]:
                        group.append(emit(o, st, "POSITION_SHIFT", now, dict(base, anchor_center=ac,
                                          current_center=oc, distance_ratio=distance, threshold=econf["position_ratio"])))
                    aq, oq = quality(anchor, econf), quality(o, econf)
                    if not preview and aq is not None and oq is not None and oq - aq + econf["float_epsilon"] >= econf["quality_gain"]:
                        group.append(emit(o, st, "QUALITY_GAIN", now, dict(base, **o["quality"],
                                          current_quality=oq, anchor_quality=aq, quality_gain=oq-aq, threshold=econf["quality_gain"])))
                st["last"] = o
                select_object(o, st, group)
        for uid in sorted(states):
            st = states[uid]
            if st["last"]["video_id"] != video:
                continue
            if st["status"] == "ACTIVE" and not preview and not partial:
                ev = emit(st["last"], st, "VIDEO_END", m["available_end_ms"], {"video_status": "completed"})
                select_object(st["last"], st, [ev])
            if not partial:
                st["status"] = "UNVERIFIED" if preview else "CLOSED"

    # Scenes are independent of object acceptance and processed in target-frame time order.
    scene_groups = defaultdict(list)
    for ev in events:
        scene_groups[ev["video_id"], ev["frame_index"]].append(ev)
    scene_times, minute_counts = defaultdict(list), Counter()
    periodic_decisions = []
    if sconf["policy"] == "hybrid":
        step = sconf["periodic_scene_ms"]
        # Reserve all periodic scenes before allowing any event scene to consume capacity.
        for video, m in sorted(videos.items()):
            candidates = [f for f in frames if f["video_id"] == video]
            first = ((m["available_start_ms"] + step - 1) // step) * step
            for requested in range(first, m["available_end_ms"], step):
                f = min(candidates, key=lambda row: (abs(row["timestamp_ms"]-requested), row["timestamp_ms"], row["frame_index"]), default=None)
                decision = dict(video_id=video, requested_timestamp_ms=requested, reason="periodic_scene", item_id=None)
                if f is None or abs(f["timestamp_ms"]-requested) > 100:
                    decision["status"] = "NO_CANDIDATE"
                elif f["decode_status"] == "failed":
                    decision["status"] = "DECODE_FAILED"
                else:
                    t = f["timestamp_ms"]
                    key, bucket = (video, f["frame_index"]), (video, t // 60000)
                    decision.update(timestamp_ms=t, frame_index=f["frame_index"])
                    if excluded(video, t):
                        decision["status"] = "QUERY_WINDOW"
                    elif key in scenes:
                        decision.update(status="MERGED", item_id=scenes[key]["item_id"])
                    elif any(abs(t-other) < sconf["min_scene_gap_ms"] for other in scene_times[video]):
                        decision["status"] = "INTERVAL"
                    elif minute_counts[bucket] >= sconf["max_scene_per_minute"]:
                        decision["status"] = "CAP"
                    else:
                        item = sample(f, "scene")
                        item["reasons"] = ["periodic_scene"]
                        scenes[key] = item
                        scene_times[video].append(t)
                        minute_counts[bucket] += 1
                        decision.update(status="ACCEPTED", item_id=item["item_id"])
                periodic_decisions.append(decision)
    for key, group in sorted(scene_groups.items(), key=lambda kv: (kv[0][0], kv[1][0]["event_timestamp_ms"], kv[0][1])):
        ev = group[0]
        v, t = ev["video_id"], ev["event_timestamp_ms"]
        bucket = (v, t // 60000)
        item = None
        if excluded(v, t):
            status = "QUERY_WINDOW"
        elif key in scenes:
            status, item = "MERGED", scenes[key]
            link(item, group)
        elif any(abs(t-other) < sconf["min_scene_gap_ms"] for other in scene_times[v]):
            status = "INTERVAL"
        elif minute_counts[bucket] >= sconf["max_scene_per_minute"]:
            status = "CAP"
        else:
            status = "ACCEPTED"
            item = sample(dict(ev, timestamp_ms=t), "scene")
            link(item, group)
            scenes[key] = item
            scene_times[v].append(t)
            minute_counts[bucket] += 1
        for i, event in enumerate(sorted(group, key=lambda x: (x["detected_at_ms"], EVENT_ORDER[x["event_type"]], x["track_uid"]))):
            event["scene_status"] = "MERGED" if item and i > 0 else status
            event["scene_item_id"] = item["item_id"] if item else None
            event["scene_limits"] = {"min_gap_ms": sconf["min_scene_gap_ms"], "max_per_minute": sconf["max_scene_per_minute"]}
    events.sort(key=lambda x: (x["video_id"], x["event_timestamp_ms"], x["track_uid"], EVENT_ORDER[x["event_type"]], x["detected_at_ms"]))
    samples = sorted([*objects.values(), *scenes.values()], key=lambda x: (
        x["video_id"], x["timestamp_ms"], x["representation"], x["track_uid"] or ""))
    summary = dict(schema_version="cctv-3.0", status="success", source_run_id=run_id,
                   policy=sconf["policy"], config_hash=config_hash,
                   counts={"events": len(events), "object": len(objects), "scene": len(scenes)},
                   event_types=dict(sorted(Counter(ev["event_type"] for ev in events).items())),
                   object_event_statuses=dict(sorted(Counter(ev["object_status"] for ev in events).items())),
                   scene_event_statuses=dict(sorted(Counter(ev["scene_status"] for ev in events).items())),
                   missing_quality_observations=sum(quality(o, econf) is None for o in tracks),
                   track_final_states={uid: st["status"] for uid, st in sorted(states.items())})
    if sconf["policy"] == "hybrid":
        summary.update(periodic_scene_ms=sconf["periodic_scene_ms"], periodic_decisions=periodic_decisions,
                       periodic_statuses=dict(sorted(Counter(d["status"] for d in periodic_decisions).items())))
    if preview:
        summary.update(execution_mode="annotation_preview", final_evaluation_ready=False,
                       disabled_events=["QUALITY_GAIN", "TRACK_GAP", "TRACK_RESUME", "VIDEO_END"])
        for row in events + samples:
            row.update(preview_only=True, timestamp_verified=False, query_exclusion_status="not_provided")
    if partial:
        summary.update(execution_mode="observed_partial", final_evaluation_ready=False,
                       disabled_events=["VIDEO_END"], completion_status="pending",
                       query_exclusion_status=config.get("query_exclusion_status", "not_provided"))
        for row in events + samples:
            row.update(provisional=True, timestamp_source="B_frames_pts", media_verified=False,
                       query_exclusion_status=config.get("query_exclusion_status", "not_provided"))
    return events, samples, summary
