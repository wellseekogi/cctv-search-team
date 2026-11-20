"""Pinned C field/crop checks; no image decoding or embedding.

Source: wellseekogi/cctv-search, commit 1a6800865e1392fbae294b20aa0fbd5356e89c07.
Extracted ItemError, integer, validate_sample and crop_box without behavior changes.
"""
import math
SCHEMA = "cctv-3.0"

class ItemError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code

def integer(value, name):
    if type(value) is not int or value < 0:
        raise ItemError("INVALID_INPUT", name + " must be a nonnegative integer")
    return value

def validate_sample(sample, video):
    required = ("item_id", "video_id", "frame_index", "timestamp_ms", "representation",
                "track_uid", "bbox", "event_ids", "reasons", "policy", "source_run_id", "track_source_id")
    if any(key not in sample for key in required) or sample.get("schema_version") != SCHEMA:
        raise ItemError("INVALID_INPUT", "Missing sample fields or incompatible schema")
    for key in ("item_id", "video_id", "policy", "source_run_id", "track_source_id"):
        if not isinstance(sample[key], str) or not sample[key]:
            raise ItemError("INVALID_INPUT", key + " must be a nonempty string")
    for key in ("event_ids", "reasons"):
        if not isinstance(sample[key], list) or any(not isinstance(x, str) for x in sample[key]):
            raise ItemError("INVALID_INPUT", key + " must be a string list")
    integer(sample["frame_index"], "frame_index")
    stamp = integer(sample["timestamp_ms"], "timestamp_ms")
    if not video["available_start_ms"] <= stamp < video["available_end_ms"]:
        raise ItemError("OUT_OF_RANGE", "Sample timestamp outside available original interval")
    if sample["representation"] == "scene":
        if sample["track_uid"] is not None or sample["bbox"] is not None:
            raise ItemError("INVALID_INPUT", "Scene must have null track_uid and bbox")
    elif sample["representation"] == "object":
        if not isinstance(sample["track_uid"], str) or not sample["track_uid"]:
            raise ItemError("INVALID_INPUT", "Object requires track_uid")
        crop_box(sample["bbox"], video["width"], video["height"])
    else:
        raise ItemError("INVALID_INPUT", "Unknown representation")

def crop_box(bbox, width, height, margin=0.1):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must contain four coordinates")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) for v in bbox):
        raise ValueError("bbox coordinates must be finite numbers")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("bbox outside original frame or zero area")
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
            min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]
