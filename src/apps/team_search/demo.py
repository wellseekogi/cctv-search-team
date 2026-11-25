"""Build the MEVA dev example from the source recording and its annotated person track."""
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

import av

from embedding.contracts import SCHEMA, file_sha256, write_jsonl

ROOT = Path(__file__).resolve().parents[3]
VIDEO_ID = "2018-03-15.14-50-00.14-55-00.school.G421"
TRACK_ID = "m0222"
FIRST_FRAME, END_FRAME, FPS = 7752, 7872, 30
DEMO_NAME = "MEVA G421 · 카페의 보행자"
DEMO_KEY = "meva-g421-m0222-v1"


def recording():
    with (ROOT / "data/meva/data_inventory.csv").open(encoding="utf-8-sig", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["video_id"] == VIDEO_ID and r["split"] == "dev")
    cache = ROOT / ".cache/meva-demo/G421.avi"
    if cache.is_file() and file_sha256(cache) == row["file_sha256"]:
        return cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=cache.parent, delete=False) as f:
        partial = Path(f.name)
    try:
        with urlopen(row["source_url"], timeout=90) as response, partial.open("wb") as f:
            shutil.copyfileobj(response, f, length=1024 * 1024)
        if file_sha256(partial) != row["file_sha256"]:
            raise ValueError("MEVA 원본 SHA-256이 팀 데이터 목록과 일치하지 않습니다.")
        os.replace(partial, cache)
    finally:
        partial.unlink(missing_ok=True)
    return cache


def make_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "G421-m0222.mp4"
    boxes = json.loads((ROOT / "fixtures/meva_person/m0222.json").read_text(encoding="utf-8"))
    first_ms = FIRST_FRAME * 1000 // FPS
    end_ms = END_FRAME * 1000 // FPS
    frames, tracks = [], []
    width, height = 960, 536
    written = 0
    with av.open(str(recording())) as src, av.open(str(path), "w") as out:
        original = src.streams.video[0]
        original.thread_type = "AUTO"
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        # F's AVI annotations use decoded frame numbers. Container PTS can arrive out of order.
        for idx, frame in enumerate(src.decode(original)):
            if idx < FIRST_FRAME:
                continue
            if idx >= END_FRAME:
                break
            local = frame.reformat(width=width, height=height, format="yuv420p")
            local.pts = None
            for packet in stream.encode(local):
                out.mux(packet)
            written += 1
            if str(idx) not in boxes:
                continue
            stamp = round(idx * 1000 / FPS)
            fields = dict(schema_version=SCHEMA, video_id=VIDEO_ID, frame_index=idx-FIRST_FRAME,
                          timestamp_ms=stamp)
            frames.append(dict(**fields, requested_timestamp_ms=stamp, time_source="pts",
                               source_offset_ms=first_ms, width=width, height=height,
                               decode_status="success", detector_status="skipped"))
            tracks.append(dict(**fields, source="gt", track_source_id="gtm1", local_track_id=TRACK_ID,
                               track_uid=f"gtm1:{VIDEO_ID}:{TRACK_ID}", class_name="person",
                               bbox=[n / 2 for n in boxes[str(idx)]], confidence=None, quality=None))
        for packet in stream.encode():
            out.mux(packet)
    if written != END_FRAME-FIRST_FRAME or len(tracks) != len(boxes):
        raise ValueError("MEVA 예제 영상 또는 사람 주석이 누락됐습니다.")
    manifest = dict(schema_version=SCHEMA, video_id=VIDEO_ID, display_name=DEMO_NAME, dataset="meva",
                    camera_id="G421", relative_path=path.name, width=width, height=height, fps=FPS,
                    duration_ms=300066, source_offset_ms=first_ms, available_start_ms=first_ms,
                    available_end_ms=end_ms, split="dev", file_sha256=file_sha256(path))
    for name, rows in (("manifest", [manifest]), ("frames", frames), ("tracks", tracks)):
        write_jsonl(directory / (name + ".jsonl"), rows)
