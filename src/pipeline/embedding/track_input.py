"""Stream B's cctv-3.0 observations into RGB crops for an image encoder."""

import gzip
import json
import math
from pathlib import Path

from .contracts import SCHEMA, check_vectors, crop_box, digest, rgb_image


def iter_jsonl(path):
    """Read one record at a time, including .jsonl.gz files."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected a JSON object")
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            yield row


def nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name + " must be a nonnegative integer")
    return value


def finite_number(value, name, minimum=0, maximum=math.inf):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(name + " is outside its finite numeric range")
    return value


def validate_track(row):
    if not isinstance(row, dict) or row.get("schema_version") != SCHEMA or row.get("source") != "pred":
        raise ValueError("Expected a cctv-3.0 pred observation")
    for key in ("track_source_id", "video_id", "local_track_id", "track_uid", "class_name"):
        if not isinstance(row.get(key), str) or not row[key]:
            raise ValueError(key + " must be a nonempty string")
    expected_uid = f"{row['track_source_id']}:{row['video_id']}:{row['local_track_id']}"
    if row["track_uid"] != expected_uid:
        raise ValueError("track_uid does not match source, video and local track ID")
    for key in ("frame_index", "timestamp_ms"):
        nonnegative_integer(row.get(key), key)
    finite_number(row.get("confidence"), "confidence", maximum=1)
    quality = row.get("quality")
    if not isinstance(quality, dict):
        raise ValueError("quality must be an object")
    finite_number(quality.get("area_ratio"), "quality.area_ratio", maximum=1)
    finite_number(quality.get("blur_var"), "quality.blur_var")
    if type(quality.get("border_touch")) is not bool:
        raise ValueError("quality.border_touch must be a boolean")
    # The handoff has no score; preserve it only when the producer supplies it.
    if "score" in quality:
        finite_number(quality["score"], "quality.score", maximum=1)
    box = row.get("bbox")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("bbox must contain four original-pixel xyxy coordinates")
    for coordinate in box:
        finite_number(coordinate, "bbox coordinate")
    if box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("bbox must have positive area")


def iter_track_batches(records, read_frame, batch_size=16, margin=0.10, min_confidence=0):
    """Yield (metadata, PIL RGB crops); never collect the full input.

    read_frame(video_id, timestamp_ms) may return B's `rgb` or C's `image_rgb`.
    Decoder, validation and model errors stop the stream instead of losing rows.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    finite_number(margin, "margin")
    finite_number(min_confidence, "min_confidence", maximum=1)
    items, images = [], []
    last_key, last_frame, last_image = None, None, None
    for position, row in enumerate(records, 1):
        try:
            validate_track(row)
            if row["confidence"] < min_confidence:
                continue
            key = (row["video_id"], row["frame_index"], row["timestamp_ms"])
            if key != last_key:
                frame = read_frame(row["video_id"], row["timestamp_ms"])
                if not isinstance(frame, dict) or frame.get("error"):
                    raise ValueError(f"Frame read failed: {frame}")
                image = rgb_image(frame["image_rgb"] if "image_rgb" in frame else frame["rgb"])
                for field in ("frame_index", "actual_timestamp_ms", "width", "height", "source_offset_ms"):
                    nonnegative_integer(frame.get(field), field)
                if (frame["frame_index"] != row["frame_index"] or
                        frame["actual_timestamp_ms"] != row["timestamp_ms"]):
                    raise ValueError("FRAME_MISMATCH: bbox requires the exact original frame")
                if image.size != (frame["width"], frame["height"]):
                    raise ValueError("FRAME_SIZE_MISMATCH")
                if not isinstance(frame.get("time_source"), str) or not frame["time_source"]:
                    raise ValueError("Missing frame time_source")
                last_key, last_frame, last_image = key, frame, image
            box = crop_box(row["bbox"], *last_image.size, margin=margin)
            item_id = "observation-" + digest([*key, row["track_uid"]])
            item = {**row, "item_id": item_id, "representation": "object", "crop_bbox": box,
                    "actual_timestamp_ms": last_frame["actual_timestamp_ms"],
                    "width": last_frame["width"], "height": last_frame["height"],
                    "source_offset_ms": last_frame["source_offset_ms"],
                    "time_source": last_frame["time_source"]}
            items.append(item)
            images.append(last_image.crop(box))
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"Observation {position}: {exc}") from exc
        if len(items) == batch_size:
            yield items, images
            items, images = [], []
    if items:
        yield items, images


def iter_embedding_batches(records, read_frame, encoder, **options):
    """Yield aligned (metadata, float32 L2 vectors) for CLIP/SigLIP2 encoders."""
    for items, images in iter_track_batches(records, read_frame, **options):
        vectors = encoder.encode_image(images)
        check_vectors(vectors, len(items), encoder.dim)
        yield items, vectors


class SequentialVideoReader:
    """Decode one video's ordered frames once, checking B's frames.jsonl and PTS."""

    def __init__(self, video_path, video, frames):
        self.video_path = Path(video_path)
        self.video = video
        self.frames = iter(frames)
        self.container = None
        self.current_metadata = None
        self.current_frame = None
        self.first_pts = None
        self.previous_index = -1
        self.previous_timestamp = -1
        for name in ("width", "height", "source_offset_ms", "available_start_ms", "available_end_ms"):
            nonnegative_integer(video[name], name)
        if not (video["width"] > 0 and video["height"] > 0 and
                video["source_offset_ms"] <= video["available_start_ms"] < video["available_end_ms"]):
            raise ValueError("Invalid manifest dimensions or available interval")

    def __enter__(self):
        import av

        self.container = av.open(str(self.video_path))
        try:
            self.stream = self.container.streams.video[0]
            self.decoded = enumerate(self.container.decode(self.stream))
        except Exception:
            self.container.close()
            self.container = None
            raise
        return self

    def __exit__(self, *_):
        self.container.close()
        self.container = None
        close = getattr(self.frames, "close", None)
        if close is not None:
            close()

    def _metadata(self, timestamp_ms):
        if self.current_metadata is not None and timestamp_ms < self.current_metadata["timestamp_ms"]:
            raise ValueError("Tracks must be ordered by frame; backwards requests are unsupported")
        while self.current_metadata is None or self.current_metadata["timestamp_ms"] < timestamp_ms:
            row = next(self.frames, None)
            if row is None:
                raise ValueError("FRAME_NOT_FOUND in frames.jsonl")
            for field in ("frame_index", "timestamp_ms"):
                nonnegative_integer(row.get(field), field)
            if (row.get("schema_version") != SCHEMA or row.get("video_id") != self.video["video_id"] or
                    row["frame_index"] <= self.previous_index or row["timestamp_ms"] <= self.previous_timestamp):
                raise ValueError("Invalid or unordered frames.jsonl metadata")
            self.previous_index, self.previous_timestamp = row["frame_index"], row["timestamp_ms"]
            self.current_metadata = row
        row = self.current_metadata
        if row["timestamp_ms"] != timestamp_ms or row.get("decode_status") != "success":
            raise ValueError("FRAME_NOT_FOUND or failed decode in frames.jsonl")
        for name in ("width", "height", "source_offset_ms"):
            if row.get(name) != self.video[name]:
                raise ValueError("Frame metadata differs from manifest: " + name)
        return row

    def read_frame(self, video_id, timestamp_ms):
        if self.container is None:
            raise RuntimeError("Use SequentialVideoReader as a context manager")
        nonnegative_integer(timestamp_ms, "timestamp_ms")
        if video_id != self.video["video_id"]:
            raise ValueError("VIDEO_NOT_FOUND")
        if not self.video["available_start_ms"] <= timestamp_ms < self.video["available_end_ms"]:
            raise ValueError("OUT_OF_RANGE")
        target = self._metadata(timestamp_ms)
        if self.current_frame is not None and self.current_frame["frame_index"] == target["frame_index"]:
            return self.current_frame
        for index, frame in self.decoded:
            if frame.pts is not None and frame.time_base is not None:
                pts = float(frame.pts * frame.time_base)
                if self.first_pts is None:
                    self.first_pts = pts
                stamp = self.video["source_offset_ms"] + round((pts - self.first_pts) * 1000)
                time_source = "pts"
            else:
                rate = self.stream.average_rate
                if rate is None or float(rate) <= 0:
                    raise ValueError("FPS unavailable for PTS fallback")
                stamp = self.video["source_offset_ms"] + round(index * 1000 / float(rate))
                time_source = "estimated"
            if index < target["frame_index"]:
                continue
            if index != target["frame_index"] or stamp != timestamp_ms or time_source != target.get("time_source"):
                raise ValueError("FRAME_MISMATCH between decoded pixels and B metadata")
            if (frame.width, frame.height) != (self.video["width"], self.video["height"]):
                raise ValueError("FRAME_SIZE_MISMATCH with manifest")
            self.current_frame = {"rgb": frame.to_ndarray(format="rgb24"), "frame_index": index,
                                  "actual_timestamp_ms": stamp, "width": frame.width, "height": frame.height,
                                  "source_offset_ms": self.video["source_offset_ms"], "time_source": time_source}
            return self.current_frame
        raise ValueError("FRAME_NOT_FOUND: video ended before the selected frame")
