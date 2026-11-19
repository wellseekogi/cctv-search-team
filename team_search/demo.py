"""Generate a small video with known objects and synthetic B metadata."""

from pathlib import Path

import av
from PIL import Image, ImageDraw

from embedding.contracts import SCHEMA, file_sha256, write_jsonl


def draw_car(draw, x):
    red = "#e92d38"
    draw.polygon([(x + 17, 119), (x + 32, 97), (x + 76, 97), (x + 93, 119)], fill=red)
    draw.polygon([(x + 27, 116), (x + 36, 102), (x + 53, 102), (x + 53, 116)], fill="#bedde3")
    draw.polygon([(x + 59, 102), (x + 74, 102), (x + 84, 116), (x + 59, 116)], fill="#bedde3")
    draw.rounded_rectangle((x, 117, x + 109, 144), radius=8, fill=red)
    draw.rectangle((x + 100, 122, x + 107, 128), fill="#fff1ad")
    for wheel in (x + 10, x + 78):
        draw.ellipse((wheel, 133, wheel + 20, 153), fill="#181c21")
        draw.ellipse((wheel + 5, 138, wheel + 15, 148), fill="#afb3b7")
    return [x, 97, x + 110, 154]


def draw_bicycle(draw, x):
    blue = "#1688fc"
    rear, front = (x + 22, 137), (x + 85, 137)
    seat, pedal, steering = (x + 40, 104), (x + 48, 136), (x + 72, 105)
    for cx, cy in (rear, front):
        draw.ellipse((cx - 17, cy - 17, cx + 17, cy + 17), outline="#181c21", width=4)
        draw.line((cx - 13, cy, cx + 13, cy), fill="#d0d4de", width=1)
        draw.line((cx, cy - 13, cx, cy + 13), fill="#d0d4de", width=1)
    draw.line([rear, seat, pedal, rear], fill=blue, width=5)
    draw.line([seat, steering, pedal], fill=blue, width=5)
    draw.line([front, (x + 68, 92), (x + 78, 89)], fill=blue, width=5)
    draw.line([(x + 32, 99), (x + 46, 99)], fill="#181c21", width=5)
    return [x + 4, 86, x + 103, 155]


def make_demo(directory):
    """Write the demo's MP4, manifest, frame rows and track observations."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    video_path = directory / "synthetic-traffic.mp4"
    manifest_path = directory / "manifest.jsonl"
    frames_path = directory / "frames.jsonl"
    tracks_path = directory / "tracks.jsonl"
    video_id, track_source_id = "synthetic-traffic", "synthetic-demo"
    width, height = 320, 192
    fps, frames_per_object = 2, 6
    frames, tracks = [], []

    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for object_number, (class_name, draw_object) in enumerate(
            [("car", draw_car), ("bicycle", draw_bicycle)], start=1
        ):
            local_track_id = f"t{object_number:06d}"
            for step in range(frames_per_object):
                frame_index = len(frames)
                timestamp_ms = frame_index * 1000 // fps
                image = Image.new("RGB", (width, height), "#a9bdbc")
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 80, width, height), fill="#646972")
                draw.line((0, 83, width, 83), fill="#ddddcf", width=3)
                for mark in range(0, width, 64):
                    draw.line((mark, 170, mark + 34, 170), fill="#e4dec7", width=3)
                bbox = draw_object(draw, 45 + step * 18)
                for packet in stream.encode(av.VideoFrame.from_image(image)):
                    container.mux(packet)

                frame_fields = {
                    "schema_version": SCHEMA,
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                }
                frames.append({
                    **frame_fields,
                    "requested_timestamp_ms": timestamp_ms,
                    "time_source": "pts",
                    "source_offset_ms": 0,
                    "width": width,
                    "height": height,
                    "decode_status": "success",
                    "detector_status": "not_run",
                })
                tracks.append({
                    **frame_fields,
                    "source": "pred",
                    "track_source_id": track_source_id,
                    "local_track_id": local_track_id,
                    "track_uid": f"{track_source_id}:{video_id}:{local_track_id}",
                    "class_name": class_name,
                    "bbox": bbox,
                    "confidence": 1.0,
                    "quality": {
                        "area_ratio": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height),
                        "blur_var": 100.0,
                        "border_touch": False,
                    },
                })
        for packet in stream.encode():
            container.mux(packet)

    duration_ms = len(frames) * 1000 // fps
    write_jsonl(manifest_path, [{
        "schema_version": SCHEMA,
        "video_id": video_id,
        "dataset": "synthetic-demo",
        "camera_id": "demo-camera",
        "relative_path": video_path.name,
        "width": width,
        "height": height,
        "duration_ms": duration_ms,
        "fps": fps,
        "source_offset_ms": 0,
        "available_start_ms": 0,
        "available_end_ms": duration_ms,
        "split": "dev",
        "file_sha256": file_sha256(video_path),
    }])
    write_jsonl(frames_path, frames)
    write_jsonl(tracks_path, tracks)
    return manifest_path, frames_path, tracks_path
