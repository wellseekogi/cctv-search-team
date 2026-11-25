"""B notebook inference loop for unattended runs on CPU or CUDA."""

import json
import shutil
import time

import av
from litedetr_adapter import convert_detections
import torch

MAX_PTS_ERROR_MS = 100
NMS_IOU = 0.5
CONF_THRESHOLD = 0.10


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(dets, threshold=NMS_IOU):
    kept = []
    for d in sorted(dets, key=lambda x: -x["confidence"]):
        if all(box_iou(d["bbox"], p["bbox"]) < threshold for p in kept):
            kept.append(d)
    return kept


def inspect_and_recover_jsonl(path, video_id, timestamps):
    if not path.exists():
        return 0
    n, valid_end = 0, 0
    with path.open("rb") as f:
        while True:
            raw = f.readline()
            if not raw:
                return n
            try:
                assert raw.endswith(b"\n"), "미완성 JSONL 행"
                item = json.loads(raw.decode("utf-8"))
                assert n < len(timestamps), "프레임 수 초과"
                expected = timestamps[n]
                assert item["video_id"] == video_id, "다른 video_id가 기록됨"
                assert item["requested_timestamp_ms"] == expected, "요청 timestamp 불일치"
                assert abs(int(item["actual_timestamp_ms"]) - expected) <= MAX_PTS_ERROR_MS, "PTS 오차"
                assert item.get("record_type") == "detection_experiment", "기록 유형 불일치"
                assert isinstance(item["detections"], list), "detections 형식 이상"
            except Exception as exc:
                # 비정상 종료로 마지막 한 줄만 미완성인 경우에 한해 복구합니다.
                if f.readline():
                    raise RuntimeError(f"{path}: {n + 1}행 중간 손상, 수동 확인 필요: {exc}") from exc
                # 이미 완결된 JSON인데 내용이 틀린 경우는 자동으로 삭제하지 않습니다.
                if raw.endswith(b"\n"):
                    raise RuntimeError(f"{path}: {n + 1}행 내용/ID 불일치: {exc}") from exc
                backup = path.with_name(path.name + f".incomplete_backup_{int(time.time())}")
                shutil.copy2(path, backup)
                with path.open("r+b") as w:
                    w.truncate(valid_end)
                print("불완전한 마지막 JSONL 행만 복구, 백업:", backup)
                return n
            n += 1
            valid_end = f.tell()


def iter_pts_nearest(job, start_idx):
    """순차 디코딩. 각 요청마다 '첫 PTS > 요청' 경계까지의 프레임에서 최근접 선택.
    비단조 PTS의 경우에도 원본 노트북 방식의 디코드 순서 기반 정책을 보존합니다.
    """
    from collections import deque

    path = job["path"]
    cache = deque()
    first_pts = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frame_iter = enumerate(container.decode(stream))
        exhausted = False
        for requested in job["timestamps"][start_idx:]:
            while not exhausted and not any(x[0] > requested for x in cache):
                try:
                    idx, frame = next(frame_iter)
                except StopIteration:
                    exhausted = True
                    break
                if frame.pts is not None and frame.time_base is not None:
                    pts = float(frame.pts * frame.time_base)
                    if first_pts is None:
                        first_pts = pts
                    actual = job["offset_ms"] + round((pts - first_pts) * 1000)
                    source = "pts"
                else:
                    rate = stream.average_rate
                    if rate is None or float(rate) <= 0:
                        raise RuntimeError("PTS가 없고 FPS도 없어 시각 추정 불가")
                    actual = job["offset_ms"] + round(idx * 1000 / float(rate))
                    source = "estimated"
                if actual >= requested - MAX_PTS_ERROR_MS:
                    cache.append((actual, idx, source, frame))
            cache = deque(x for x in cache if x[0] >= requested - MAX_PTS_ERROR_MS)
            if not cache:
                raise RuntimeError(f"{job['video_id']}: {requested}ms 근처 프레임 없음")
            actual, idx, source, selected = min(cache, key=lambda x: (abs(x[0] - requested), x[0]))
            if abs(actual - requested) > MAX_PTS_ERROR_MS:
                raise RuntimeError(f"{job['video_id']}: 요청 {requested}ms, 실제 {actual}ms (오차 초과)")
            yield dict(
                requested_timestamp_ms=requested,
                actual_timestamp_ms=actual,
                frame_index=idx,
                time_source=source,
                source_offset_ms=job["offset_ms"],
                rgb=selected.to_ndarray(format="rgb24"),
            )


def run_one_video(job, model, postprocessors, preprocess):
    device = next(model.parameters()).device
    video_id, outfile = job["video_id"], job["output"]
    timestamps, total = job["timestamps"], job["total"]
    print("\n영상:", video_id, "| 목표:", total, "| 출력:", outfile, flush=True)
    start = inspect_and_recover_jsonl(outfile, video_id, timestamps)
    print("이전 정상 프레임:", start, "/", total)
    if start == total:
        return
    assert not model.training
    assert "bbox" in postprocessors
    iterator = iter_pts_nearest(job, start)
    # 최소 1프레임 검사: 프레임 크기·시각 조건
    from PIL import Image

    started = time.perf_counter()
    with outfile.open("a", encoding="utf-8") as f:
        for j, fr in enumerate(iterator, start=start):
            assert fr["requested_timestamp_ms"] == timestamps[j]
            image = Image.fromarray(fr["rgb"])
            assert (image.width, image.height) == (job["width"], job["height"])
            tensor, _ = preprocess(image, None)
            tensor = tensor.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                pred = model([tensor])
                sizes = torch.tensor([[image.height, image.width]], device=device)
                result = postprocessors["bbox"](pred, sizes)[0]
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - t0
            dets = nms(convert_detections(result, image.width, image.height, CONF_THRESHOLD))
            record = dict(
                schema_version="cctv-3.0",
                record_type="detection_experiment",
                video_id=video_id,
                requested_timestamp_ms=fr["requested_timestamp_ms"],
                actual_timestamp_ms=fr["actual_timestamp_ms"],
                timestamp_ms=fr["actual_timestamp_ms"],
                frame_index=fr["frame_index"],
                width=image.width,
                height=image.height,
                time_source=fr["time_source"],
                source_offset_ms=fr["source_offset_ms"],
                inference_seconds=latency,
                detections=dets,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            del pred, result, tensor
            if (j + 1) % 25 == 0 or j + 1 == total:
                print(
                    f"{video_id} {j + 1}/{total} | {len(dets)} persons | {latency:.3f}s | "
                    f"{(time.perf_counter() - started) / 60:.1f}min",
                    flush=True,
                )
    n = inspect_and_recover_jsonl(outfile, video_id, timestamps)
    assert n == total, f"{video_id}: 불완료 {n}/{total}"
    print("PASS:", video_id, n, flush=True)
