"""C CLI: --frame-reader integrations.b_reader:read_frame.

Set CCTV_B_READER_CONFIG to a local JSON configuration file. Importing this
module does not initialize B, load video, or download models.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path


def normalize_frame(result, timestamp_ms, expected_frame_index=None):
    if not isinstance(result, dict):
        raise ValueError("B must return a dictionary")
    if result.get("error"):
        raise ValueError(str(result["error"]))
    result = dict(result)
    if "image_rgb" not in result:
        if "rgb" not in result:
            raise ValueError("B frame missing rgb/image_rgb")
        result["image_rgb"] = result["rgb"]
    for name in ("actual_timestamp_ms", "frame_index", "width", "height", "source_offset_ms", "time_source"):
        if name not in result:
            raise ValueError("B frame missing " + name)
    if result["actual_timestamp_ms"] != timestamp_ms:
        raise ValueError("FRAME_MISMATCH: decoded time differs from A selected time")
    if expected_frame_index is not None and result["frame_index"] != expected_frame_index:
        raise ValueError("FRAME_MISMATCH: decoded frame index differs from A")
    return result


def motion_frame(reader, frame):
    result = normalize_frame(reader(frame["video_id"], frame["timestamp_ms"]),
                             frame["timestamp_ms"], frame["frame_index"])
    if (result["width"], result["height"]) != (frame["width"], frame["height"]):
        raise ValueError("FRAME_SIZE_MISMATCH")
    if result["source_offset_ms"] != frame.get("source_offset_ms", 0):
        raise ValueError("TIME_METADATA_MISMATCH")
    return result["image_rgb"]


_reader = None


def configure(config_path):
    global _reader
    path = Path(config_path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    def resolve(name):
        return (path.parent / config[name]).resolve()
    manifest = resolve("manifest")
    data_root = resolve("data_root")
    overrides = {}
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        video = json.loads(line)
        source = (data_root / video["relative_path"]).resolve()
        if not source.is_relative_to(data_root):
            raise ValueError("Video path escapes data_root")
        if not source.is_file():
            raise FileNotFoundError(f"Original video required: {source}")
        sha = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                sha.update(chunk)
        if sha.hexdigest() != video["file_sha256"]:
            raise ValueError("Original video hash mismatch: " + video["video_id"])
        overrides[video["video_id"]] = source
    spec = importlib.util.spec_from_file_location("cctv_supplied_b_reader", resolve("b_reader_module"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _reader = module.CCTVFrameReader(manifest, resolve("vision_config"), path_overrides=overrides)
    return _reader


def read_frame(video_id, requested_timestamp_ms):
    global _reader
    if _reader is None:
        config = os.environ.get("CCTV_B_READER_CONFIG")
        if not config:
            raise RuntimeError("Set CCTV_B_READER_CONFIG to a configured b_reader.local.json")
        configure(config)
    return normalize_frame(_reader.read_frame(video_id, requested_timestamp_ms), requested_timestamp_ms)
