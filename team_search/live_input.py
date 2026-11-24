"""Index one completed video segment or one B handoff."""
import json
from contextlib import closing
from pathlib import Path

import av

from embedding.contracts import SCHEMA, file_sha256, read_jsonl, within, write_jsonl
from embedding.pipeline import build_index
from embedding.track_input import SequentialVideoReader
from sampling.core import run_sampling
from .__main__ import ROOT
from .demo import make_demo
from .inputs import build_index_from_team


T8_SOURCE = "b_litedetr_nom30_800_t8"


def decoded(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 0)
        first_pts = None
        previous = -200
        for index, frame in enumerate(container.decode(stream)):
            if frame.time is None and rate <= 0:
                raise ValueError("영상 프레임 시각을 읽을 수 없습니다.")
            if frame.time is not None and first_pts is None:
                first_pts = frame.time
            stamp = round(1000 * (frame.time - first_pts if frame.time is not None else index / rate))
            if stamp > 120000 or index >= 18000 or frame.width * frame.height > 16000000:
                raise ValueError("영상을 120초, 18,000프레임, 1,600만 화소 이하의 구간으로 나눠 주세요.")
            if stamp - previous >= 200:
                yield index, stamp, frame, rate
                previous = stamp


def clip_inputs(path, key, name):
    frames = []
    for index, stamp, frame, rate in decoded(path):
        frames.append(dict(schema_version=SCHEMA, video_id=key, frame_index=index, timestamp_ms=stamp,
                           requested_timestamp_ms=stamp, width=frame.width, height=frame.height,
                           decode_status="success", detector_status="skipped"))
    if not frames:
        raise ValueError("디코딩할 프레임이 없습니다.")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        length = round(float(stream.duration * stream.time_base) * 1000) if stream.duration else 0
    duration = max(length, stamp + max(1, round(1000 / (rate or 1))))
    if duration > 120000:
        raise ValueError("각 영상 구간은 최대 120초입니다.")
    manifest = [dict(schema_version=SCHEMA, video_id=key, dataset="upload", camera_id="upload",
                     relative_path=path.name, width=frame.width, height=frame.height, duration_ms=duration,
                     fps=rate or 1, source_offset_ms=0, available_start_ms=0, available_end_ms=duration,
                     split="dev", file_sha256=file_sha256(path), display_name=name)]
    return manifest, frames


def process_segment(kind, key, name, source, stage, config, encoder, *,
                    cache=ROOT / ".cache/embedding",
                    vision_config=ROOT / "modules/b_detection_tracking/configs/vision.yaml"):
    stage = Path(stage)
    work = stage / "input"
    work.mkdir()
    tracks, exclusions = [], []
    if kind == "clip":
        root = source.parent
        manifests, frames = clip_inputs(source, key, name)
        policy = "motion"
    elif kind == "demo":
        make_demo(work)
        root = work
        manifests = read_jsonl(root / "manifest.jsonl")
        frames = read_jsonl(root / "frames.jsonl")
        tracks = read_jsonl(root / "tracks.jsonl")
        policy = "hybrid"
    else:
        root = source
        manifests, frames, tracks = [read_jsonl(within(root, name + ".jsonl"))
                                     for name in ("manifest", "frames", "tracks")]
        if kind == "b_t8" and {(row["track_source_id"], row["source"], row["class_name"]) for row in tracks} != {
            (T8_SOURCE, "pred", "person")
        }:
            raise ValueError("B 객체 개선 검색에는 T8 사람 추적 인계가 필요합니다.")
        exclusion_file = within(root, "query_exclusions.jsonl")
        if exclusion_file.exists():
            exclusions = read_jsonl(exclusion_file)
        policy = "hybrid"
    if not manifests or any(row["split"] != "dev" for row in manifests):
        raise ValueError("라이브 검색에는 dev 분할 영상만 등록할 수 있습니다.")
    statuses = (read_jsonl(within(root, "video_status.jsonl")) if kind in ("handoff", "b_t8") else
                [dict(video_id=m["video_id"], available_end_ms=m["available_end_ms"],
                      video_status="completed") for m in manifests])
    settings = json.loads((ROOT / "configs/person" / (policy + ".yaml")).read_text(encoding="utf-8"))
    if not tracks:
        settings["track_source_id"] = "segment-" + key
    if kind == "clip":
        with closing(decoded(source)) as sequence:
            def load(frame):
                index, stamp, raw, _ = next(sequence)
                if (index, stamp) != (frame["frame_index"], frame["timestamp_ms"]):
                    raise ValueError("원본 프레임이 변경됐습니다.")
                return raw.to_ndarray(format="rgb24")
            events, samples, summary = run_sampling(settings, manifests, frames, tracks, statuses,
                                                    exclusions, key, frame_loader=load)
    else:
        events, samples, summary = run_sampling(settings, manifests, frames, tracks, statuses, exclusions, key)
    write_jsonl(stage / "events.jsonl", events)
    write_jsonl(stage / "samples.jsonl", samples)
    write_jsonl(stage / "manifest.jsonl", manifests)
    (stage / "selection.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    if not samples:
        raise ValueError("이 구간에서 검색에 사용할 장면이 선택되지 않았습니다.")
    if kind == "b_t8":
        if len(manifests) != 1:
            raise ValueError("T8 인계는 한 영상씩 등록하세요.")
        video = manifests[0]
        path = within(root, video["relative_path"])
        ordered = sorted(frames, key=lambda f: f["timestamp_ms"])
        with SequentialVideoReader(path, video, ordered) as reader:
            def read_selected(vid, stamp):
                frame = reader.read_frame(vid, stamp)
                return dict(frame, image_rgb=frame["rgb"])

            batch = config["batch_size_gpu" if encoder.device.startswith("cuda") else "batch_size_cpu"]
            status = build_index(stage / "samples.jsonl", stage / "manifest.jsonl", root, stage / "index",
                                 cache, read_selected, encoder, "b-cctv-frame-reader-pyav-v1",
                                 config["crop_margin_ratio"], batch)
    else:
        status = build_index_from_team(manifest=stage / "manifest.jsonl", data_root=root,
                                       output=stage / "index", cache=cache,
                                       embedding_config=config, encoder=encoder,
                                       vision_config=vision_config,
                                       samples=stage / "samples.jsonl")
    if status["status"] != "success":
        raise ValueError("임베딩 실패 " + str(read_jsonl(stage / "index/errors.jsonl")[:1]))
    count = len(events) if policy != "motion" else summary["motion_candidate_count"]
    return count, len(samples)
