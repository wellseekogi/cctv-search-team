"""Prepare a live T8 handoff for an MP4 excerpt of one of the attached MEVA originals."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from zipfile import ZipFile

import av
import numpy as np

from embedding.contracts import file_sha256, write_jsonl
from team_search.live_input import T8_SOURCE, clip_inputs


def signature_frames(path, indices):
    last = max(indices)
    images = {}
    with av.open(str(path)) as video:
        for i, frame in enumerate(video.decode(video=0)):
            if i in indices:
                images[i] = frame.reformat(width=160, height=90, format="gray").to_ndarray().astype(np.float32)
            if i == last:
                break
    if set(images) != indices:
        raise ValueError(f"프레임을 모두 읽을 수 없습니다: {path.name}")
    return images


def read_observations(archive, original_sha):
    with ZipFile(archive) as bundle:
        reports = [(entry, json.loads(bundle.read(entry))) for entry in bundle.namelist()
                   if entry.endswith("/validation_report.json")]
        matches = [(entry, report) for entry, report in reports
                   if report["input_sha256"]["video_file"] == original_sha]
        if len(matches) != 1:
            raise ValueError("T8 인계에서 원본 영상 해시와 일치하는 카메라 하나가 필요합니다.")
        entry, report = matches[0]
        folder = entry.rsplit("/", 1)[0]
        rows = []
        for name in ("frames.jsonl", "tracks.jsonl"):
            payload = bundle.read(f"{folder}/{name}")
            if hashlib.sha256(payload).hexdigest() != report["files_sha256"][name]:
                raise ValueError(f"T8 인계 파일 해시 불일치: {name}")
            rows.append([json.loads(line) for line in payload.splitlines() if line.strip()])
        frames, tracks = rows
        if (len(frames) != report["sampled_frames"] or len(tracks) != report["tracks_observations"]
                or {r["track_source_id"] for r in tracks} != {T8_SOURCE}
                or {r["video_id"] for r in frames + tracks} != {report["video_id"]}):
            raise ValueError("T8 인계 영상 또는 관측 수가 검증 보고서와 다릅니다.")
        return frames, tracks


def prepare(archive, original, clip, start, output):
    key = file_sha256(clip)
    original_sha = file_sha256(original)
    manifest_rows, frames = clip_inputs(clip, key, clip.name)
    manifest = manifest_rows[0]
    if start < 0 or start % 200 or manifest["fps"] != 30:
        raise ValueError("T8 인계는 30fps 영상과 200ms 배수의 원본 시작 시각이 필요합니다.")
    start_index = start * 30 // 1000
    checkpoints = {frames[0]["frame_index"], frames[len(frames) // 2]["frame_index"], frames[-1]["frame_index"]}
    input_images = signature_frames(original, {i + start_index for i in checkpoints})
    excerpt_images = signature_frames(clip, checkpoints)
    if any(np.abs(input_images[i + start_index] - excerpt_images[i]).mean() > 3 for i in checkpoints):
        raise ValueError("MP4 내용이 입력한 원본 시각과 일치하지 않습니다.")

    t8_frames, tracks = read_observations(archive, original_sha)
    stamps = {f["timestamp_ms"] + start: f for f in frames}
    selected = [r for r in t8_frames if r["timestamp_ms"] in stamps]
    if (len(selected) != len(frames) or any(r["frame_index"] - start_index != stamps[r["timestamp_ms"]]["frame_index"]
                                             or r["decode_status"] != "success" for r in selected)):
        raise ValueError("T8 관측과 MP4의 프레임/시각이 일치하지 않습니다.")
    if any((r["width"], r["height"]) != (selected[0]["width"], selected[0]["height"]) for r in selected):
        raise ValueError("선택한 T8 구간의 원본 크기가 일정하지 않습니다.")
    scale_x, scale_y = manifest["width"] / selected[0]["width"], manifest["height"] / selected[0]["height"]
    tracks = [dict(r, video_id=key, frame_index=r["frame_index"] - start_index,
                   timestamp_ms=r["timestamp_ms"] - start,
                   track_uid=f'{T8_SOURCE}:{key}:{r["local_track_id"]}',
                   bbox=[p * scale for p, scale in zip(r["bbox"], (scale_x, scale_y, scale_x, scale_y), strict=True)])
              for r in tracks if r["timestamp_ms"] in stamps]
    if not tracks:
        raise ValueError("선택한 구간에 T8 사람 추적 결과가 없습니다.")
    manifest.update(b_source_sha256=original_sha, b_source_start_ms=start)
    statuses = [dict(video_id=key, video_status="completed", available_end_ms=manifest["available_end_ms"])]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / "handoff"
        stage.mkdir()
        observed_frames = [dict(f, detector_status="ran", source_offset_ms=0, time_source="pts") for f in frames]
        for name, rows in (("manifest", [manifest]), ("frames", observed_frames),
                           ("tracks", tracks), ("video_status", statuses)):
            write_jsonl(stage / (name + ".jsonl"), rows)
        shutil.copyfile(clip, stage / clip.name)
        stage.rename(output)
    print(f"T8: {len(frames)} frames, {len(tracks)} observations ({output})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "original", "clip", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--start-ms", type=int, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.original, args.clip, args.start_ms, args.output)
