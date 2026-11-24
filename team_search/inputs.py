import tempfile
from pathlib import Path

from embedding import build_index, create_encoder
from embedding.contracts import SCHEMA, digest, file_sha256, read_jsonl, within, write_jsonl
from embedding.track_input import iter_jsonl, nonnegative_integer, validate_track
from modules.b_detection_tracking.src.b_video_io import CCTVFrameReader


def _frame_key(row):
    if row.get("schema_version") != SCHEMA or not isinstance(row.get("video_id"), str) or not row["video_id"]:
        raise ValueError("Expected a cctv-3.0 B record with a video_id")
    return (row["video_id"], nonnegative_integer(row.get("frame_index"), "frame_index"),
            nonnegative_integer(row.get("timestamp_ms"), "timestamp_ms"))


def _b_samples(frames, tracks, source_run_id):
    if source_run_id is None:
        source_run_id = "b-export-" + digest([file_sha256(path) for path in (frames, tracks) if path])
    successful_frames = set()
    if frames:
        for frame in iter_jsonl(frames):
            key = _frame_key(frame)
            if frame.get("decode_status") == "failed":
                continue
            if frame.get("decode_status") != "success" or key in successful_frames:
                raise ValueError(f"Invalid decode status or duplicate frame in B export: {key}")
            successful_frames.add(key)
            run_id = frame.get("source_run_id", source_run_id)
            yield {
                "schema_version": SCHEMA, "item_id": "b-scene-" + digest([run_id, key]),
                "video_id": key[0], "frame_index": key[1], "timestamp_ms": key[2],
                "representation": "scene", "track_uid": None, "bbox": None,
                "event_ids": [], "reasons": ["b_decoded_frame"], "policy": "b_export",
                "track_source_id": frame.get("track_source_id", "b-frame"), "source_run_id": run_id,
            }
    if tracks:
        for track in iter_jsonl(tracks):
            validate_track(track)
            key = _frame_key(track)
            if frames and key not in successful_frames:
                raise ValueError(f"B track has no matching successfully decoded frame: {key}")
            run_id = track.get("source_run_id", source_run_id)
            yield {
                **track, "item_id": "b-object-" + digest([run_id, key, track["track_uid"]]),
                "representation": "object", "event_ids": [], "reasons": ["b_track_observation"],
                "policy": "b_export", "source_run_id": run_id,
            }


def _selected_manifest(manifest, samples):
    videos = {}
    for row in read_jsonl(manifest):
        video_id = row.get("video_id")
        if not isinstance(video_id, str) or not video_id or video_id in videos:
            raise ValueError(f"Invalid or duplicate manifest video_id: {video_id}")
        # The B T3 handoff used the current fields before it recorded the schema version.
        if row.get("schema_version", SCHEMA) != SCHEMA:
            raise ValueError(f"Unsupported manifest schema for {video_id}")
        videos[video_id] = {"schema_version": SCHEMA, **row}
    selected = set()
    for sample in iter_jsonl(samples):
        video_id = sample.get("video_id")
        if not isinstance(video_id, str) or video_id not in videos:
            raise ValueError(f"Sample video_id missing from manifest: {video_id}")
        selected.add(video_id)
    return [row for video_id, row in videos.items() if video_id in selected]


def build_index_from_team(*, manifest, data_root, output, cache, embedding_config, vision_config,
                          samples=None, frames=None, tracks=None, source_run_id=None, encoder=None):
    """Build C's query index from A's unchanged samples or B exports."""
    if (samples is None) == (frames is None and tracks is None):
        raise ValueError("Supply --samples or at least one of --frames/--tracks")

    with tempfile.TemporaryDirectory(prefix="cctv-inputs-") as temp:
        if samples is None:
            samples = Path(temp) / "samples.jsonl"
            write_jsonl(samples, _b_samples(frames, tracks, source_run_id))
        videos = _selected_manifest(manifest, samples)
        manifest = Path(temp) / "manifest.jsonl"
        write_jsonl(manifest, videos)
        paths = {row["video_id"]: within(data_root, row["relative_path"]) for row in videos}
        reader = CCTVFrameReader(manifest, vision_config, path_overrides=paths)

        if encoder is None:
            encoder = create_encoder(embedding_config)
        batch_key = "batch_size_gpu" if encoder.device.startswith("cuda") else "batch_size_cpu"
        with reader.ordered() as read:
            def read_frame(video_id, timestamp_ms):
                frame = read(video_id, timestamp_ms)
                return {**frame, "image_rgb": frame["rgb"]}

            return build_index(samples, manifest, data_root, output, cache, read_frame, encoder,
                               "b-cctv-frame-reader-pyav-v1", embedding_config["crop_margin_ratio"],
                               embedding_config[batch_key])
