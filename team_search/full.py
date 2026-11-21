"""Build the interactive gallery from B observations, A hybrid selection and C embeddings."""

import json
import os
import tempfile
from pathlib import Path

import av
from filelock import FileLock

from embedding.contracts import digest, file_sha256, read_jsonl, within, write_jsonl
from embedding.pipeline import build_index
from embedding.track_input import SequentialVideoReader
from integrations.f_metadata import align_manifest, image_exclusions
from rag.engine import FrameRecommender
from sampling.core import run_sampling
from .baseline import ensure_observations, source_video

ROOT = Path(__file__).resolve().parents[1]


def add_full_options(parser):
    parser.add_argument(
        "--metadata-root", type=Path, default=ROOT / "data", help="MEVA·VIRAT manifest와 질의 목록 폴더"
    )
    parser.add_argument(
        "--b-inputs", type=Path, default=Path(os.getenv("CCTV_B_INPUTS", ROOT / "runs/b-inputs"))
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.getenv("CCTV_DATA_ROOT", ROOT / "runs/source-data"))
    )
    parser.add_argument("--gallery-cache", type=Path, default=ROOT / "runs/a-full")


def observation_folders(root):
    """Identify native B exports and handoffs by source hash, including B/F ID aliases."""
    folders = {}
    for path in sorted(root.rglob("frames.jsonl")):
        folder = path.parent
        report = folder / "validation_report.json"
        if report.is_file():
            data = json.loads(report.read_text(encoding="utf-8"))
            sha = data["input_sha256"]["video_file"]
        else:
            manifest = folder / "manifest.jsonl"
            if not manifest.is_file():
                manifest = folder / "manifest.PROVISIONAL.jsonl"
            rows = read_jsonl(manifest)
            if len(rows) != 1:
                raise ValueError(f"영상별 B manifest가 필요합니다: {folder}")
            sha = rows[0]["file_sha256"]
        if sha in folders:
            raise ValueError(f"같은 원본의 B 출력이 두 곳에 있습니다: {folders[sha]}, {folder}")
        folders[sha] = folder
    return folders


def complete_grid(frames, video):
    expected = range(video["available_start_ms"], video["available_end_ms"], 200)
    return sorted(r["requested_timestamp_ms"] for r in frames) == list(expected)


def build_video(folder, original, data_root, queries, settings, config, encoder, stage):
    frames = sorted(read_jsonl(folder / "frames.jsonl"), key=lambda r: r["requested_timestamp_ms"])
    tracks = read_jsonl(folder / "tracks.jsonl")
    if not complete_grid(frames, original):
        raise ValueError(f"{original['video_id']}: B 관측이 전체 구간을 덮지 않습니다 ({len(frames)}행).")
    ref = frames[0]
    video = align_manifest(dict(ref, file_sha256=original["file_sha256"]), [original], frames)
    report_path = folder / "validation_report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["video_id"] != video["video_id"] or report["sampled_frames"] != len(frames):
            raise ValueError(f"B 검증 보고서와 관측이 다릅니다: {folder}")
        for name in ("frames.jsonl", "tracks.jsonl"):
            if file_sha256(folder / name) != report["files_sha256"][name]:
                raise ValueError(f"B 검증 해시가 다릅니다: {folder / name}")
    status_file = folder / "video_status.jsonl"
    if status_file.is_file():
        statuses = read_jsonl(status_file)
    elif report_path.is_file():
        statuses = [
            dict(
                video_id=video["video_id"],
                video_status="completed",
                available_end_ms=video["available_end_ms"],
            )
        ]
    else:
        raise ValueError(f"B 완료 상태 또는 검증 보고서가 필요합니다: {folder}")
    if any(t["source"] != "pred" for t in tracks):
        raise ValueError("전체 검색에는 B 예측 트랙이 필요합니다.")
    run_id = "a-hybrid-" + video["video_id"]
    settings = dict(settings, track_source_id=tracks[0]["track_source_id"] if tracks else "b-no-tracks")
    events, samples, _ = run_sampling(
        settings, [video], frames, tracks, statuses, image_exclusions(queries, video), run_id
    )
    write_jsonl(stage / "events.jsonl", events)
    write_jsonl(stage / "samples.jsonl", samples)
    write_jsonl(stage / "manifest.jsonl", [video])
    with SequentialVideoReader(within(data_root, video["relative_path"]), video, frames) as reader:

        def read_selected(video_id, stamp):
            frame = reader.read_frame(video_id, stamp)
            return dict(frame, image_rgb=frame["rgb"])

        batch = config["batch_size_gpu" if encoder.device.startswith("cuda") else "batch_size_cpu"]
        status = build_index(
            stage / "samples.jsonl",
            stage / "manifest.jsonl",
            data_root,
            stage / "index",
            stage.parent / "cache",
            read_selected,
            encoder,
            "b-sequential-pyav-v1",
            config["crop_margin_ratio"],
            batch,
        )
    if status["status"] != "success":
        raise ValueError(f"C 임베딩 실패: {video['video_id']} {read_jsonl(stage / 'index/errors.jsonl')[:1]}")


def full_catalog(args, config, encoder):
    metadata = args.metadata_root
    settings_path = ROOT / "configs/person/hybrid.yaml"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    videos, queries = [], []
    for dataset in ("meva", "virat"):
        videos.extend(read_jsonl(metadata / dataset / "manifest.jsonl"))
        queries.extend(read_jsonl(metadata / dataset / "queries.jsonl"))
    folders = observation_folders(args.b_inputs)
    catalog = {}
    args.gallery_cache.mkdir(parents=True, exist_ok=True)
    for original in videos:
        path = source_video(metadata, args.data_root, original)
        folder = folders.get(original["file_sha256"])
        if folder and not complete_grid(read_jsonl(folder / "frames.jsonl"), original):
            print(f"부분 B 인계 대신 전체 구간을 처리합니다: {original['video_id']}", flush=True)
            folder = None
        if folder is None:
            folder = ensure_observations(original, path, args.gallery_cache, encoder.device)
        files = {
            n: file_sha256(folder / n)
            for n in ("frames.jsonl", "tracks.jsonl", "video_status.jsonl", "validation_report.json")
            if (folder / n).is_file()
        }
        key = "a-" + digest(
            [
                original,
                files,
                settings,
                queries,
                encoder.metadata,
                config["crop_margin_ratio"],
                file_sha256(Path(__file__)),
                file_sha256(ROOT / "sampling/core.py"),
            ]
        )
        final = args.gallery_cache / key
        with FileLock(str(final) + ".lock"):
            if not final.exists():
                print(f"A 선별 · C 임베딩: {original['video_id']}", flush=True)
                with tempfile.TemporaryDirectory(dir=args.gallery_cache) as temp:
                    stage = Path(temp) / "video"
                    stage.mkdir()
                    build_video(folder, original, args.data_root, queries, settings, config, encoder, stage)
                    stage.rename(final)
        engine = FrameRecommender(final / "index", encoder)
        video = read_jsonl(final / "manifest.jsonl")[0]
        catalog[key] = engine, {video["video_id"]: (path, video["available_end_ms"], video["video_id"])}
    return catalog


def playback_file(source, cache):
    target = cache / "playback.mp4"
    with FileLock(str(cache / "playback.lock")):
        if target.exists():
            return target
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            pending = Path(temp) / target.name
            with av.open(str(source)) as src:
                stream = src.streams.video[0]
                h264 = stream.codec_context.name == "h264"
                if h264 and source.suffix.lower() == ".mp4":
                    return source
                with av.open(str(pending), "w", options={"movflags": "+faststart"}) as dst:
                    if h264:
                        out = dst.add_stream_from_template(stream)
                        first_pts = None
                        for packet in src.demux(stream):
                            if packet.dts is None or packet.pts is None:
                                continue
                            if first_pts is None:
                                first_pts = packet.pts
                            packet.pts -= first_pts
                            packet.dts -= first_pts
                            packet.stream = out
                            dst.mux(packet)
                    else:
                        out = dst.add_stream("libx264", rate=stream.average_rate)
                        out.width, out.height = stream.width, stream.height
                        out.pix_fmt = "yuv420p"
                        out.options = {"preset": "veryfast"}
                        first_pts = None
                        for frame in src.decode(stream):
                            if first_pts is None:
                                first_pts = frame.pts
                            frame.pts -= first_pts
                            for packet in out.encode(frame):
                                dst.mux(packet)
                        for packet in out.encode():
                            dst.mux(packet)
            pending.rename(target)
    return target
