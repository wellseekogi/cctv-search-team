"""validate: B 산출물과 F 평가 입력을 검사한다."""
from collections import Counter
from pathlib import Path

from .core import UserError, file_sha256, read_jsonl, within


def load_inputs(folder, start_ms=None, end_ms=None):
    folder = Path(folder)
    if not folder.is_dir():
        raise UserError(f"입력 폴더가 없습니다: {folder}")

    def rows(name, required=True):
        path = folder / name
        if not path.is_file():
            if required:
                raise UserError(f"입력 파일이 없습니다: {path}")
            return []
        return read_jsonl(path)

    manifests, frames, tracks, statuses = (rows(n) for n in (
        "manifest.jsonl", "frames.jsonl", "tracks.jsonl", "video_status.jsonl"))
    exclusions = rows("query_exclusions.jsonl", required=False)
    if start_ms is not None or end_ms is not None:
        def inside(row):
            t = row["timestamp_ms"]
            return (start_ms is None or t >= start_ms) and (end_ms is None or t < end_ms)
        frames, tracks = [f for f in frames if inside(f)], [t for t in tracks if inside(t)]
        if not frames:
            raise UserError("지정한 시간 구간에 프레임이 없습니다.")
    return manifests, frames, tracks, statuses, exclusions


def validate_observations(config, data, folder, check_media, check_hash, require_split, windowed):
    from sampling.core import InputError, validate as core_validate
    manifests, frames, tracks, statuses, exclusions = data
    try:
        core_validate(config, manifests, frames, tracks, statuses, exclusions)
    except (InputError, KeyError, TypeError) as exc:
        raise UserError(f"입력 검증 실패: {exc}") from exc
    splits = {m.get("split") for m in manifests}
    if require_split and (len(splits) != 1 or not splits <= {"dev", "eval"}):
        raise UserError(f"manifest의 split은 dev 또는 eval 하나여야 합니다: {sorted(map(str, splits))}")
    if check_media:
        for m in manifests:
            path = within(folder, m["relative_path"])
            if not path.is_file():
                raise UserError(f"영상 파일이 없습니다: {path}")
            if check_hash and file_sha256(path) != m["file_sha256"]:
                raise UserError(f"영상 SHA-256이 manifest와 다릅니다: {path.name}")
    last = {}
    for f in frames:
        last[f["video_id"]] = max(last.get(f["video_id"], 0), f["timestamp_ms"])
    warnings = [] if windowed else [
        f"{m['video_id']}: 마지막 프레임 {last.get(m['video_id'], 0)}ms < 영상 {m['available_end_ms']}ms "
        "(B 처리가 일부만 된 것일 수 있음)"
        for m in manifests if last.get(m["video_id"], 0) < 0.9 * m["available_end_ms"]]
    return dict(videos=len(manifests), splits=sorted(map(str, splits)), frames=len(frames), tracks=len(tracks),
                classes=dict(Counter(t["class_name"] for t in tracks)),
                track_sources=sorted({t["track_source_id"] for t in tracks}), warnings=warnings)


def validate_eval_fixture(folder):
    """F의 평가 입력(manifest·GT·질의·정답·대상·예측)의 ID·시각·박스·참조를 검사한다."""
    folder = Path(folder)
    errors = []

    def err(message):
        if len(errors) < 30:
            errors.append(message)

    def rows(name, required=True):
        path = folder / name
        if not path.is_file():
            if required:
                err(f"{name}: 파일이 없습니다")
            return []
        return read_jsonl(path)

    manifest, gt = rows("manifest.jsonl"), rows("tracks_gt.jsonl")
    queries, qrels = rows("queries.jsonl"), rows("qrels.jsonl")
    targets, predictions = rows("targets.jsonl"), rows("predictions.jsonl", False)

    videos = {}
    for i, m in enumerate(manifest, 1):
        v = m.get("video_id")
        if not isinstance(v, str) or not v or v in videos:
            err(f"manifest.jsonl:{i}: video_id가 없거나 중복")
            continue
        keys = ("width", "height", "duration_ms", "available_start_ms", "available_end_ms")
        if not all(type(m.get(k)) is int for k in keys) or not (
                m["width"] > 0 and m["height"] > 0 and 0 <= m["available_start_ms"] < m["available_end_ms"] <= m["duration_ms"]):
            err(f"manifest.jsonl:{i}: 크기·시간 필드가 올바르지 않음")
            continue
        if m.get("split") not in ("dev", "eval"):
            err(f"manifest.jsonl:{i}: split은 dev 또는 eval")
        if m.get("schema_version") != "cctv-3.0":
            err(f"manifest.jsonl:{i}: schema_version은 cctv-3.0")
        videos[v] = m

    seen, uids = set(), set()
    for i, r in enumerate(gt, 1):
        m = videos.get(r.get("video_id"))
        if m is None:
            err(f"tracks_gt.jsonl:{i}: manifest에 없는 영상 {r.get('video_id')}")
            continue
        t, b, uid = r.get("timestamp_ms"), r.get("bbox"), r.get("track_uid")
        if type(t) is not int or not 0 <= t < m["duration_ms"]:
            err(f"tracks_gt.jsonl:{i}: timestamp_ms가 영상 길이 밖")
        if not (isinstance(b, list) and len(b) == 4
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in b)
                and 0 <= b[0] < b[2] <= m["width"] and 0 <= b[1] < b[3] <= m["height"]):
            err(f"tracks_gt.jsonl:{i}: bbox가 원본 범위를 벗어남 {b}")
        key = (r.get("video_id"), uid, t)
        if key in seen:
            err(f"tracks_gt.jsonl:{i}: 같은 영상·트랙·시각 중복")
        seen.add(key)
        uids.add(uid)

    query_ids = set()
    for i, q in enumerate(queries, 1):
        qid = q.get("query_id")
        if not qid or qid in query_ids:
            err(f"queries.jsonl:{i}: query_id가 없거나 중복")
        query_ids.add(qid)
        if q.get("split") not in ("dev", "eval"):
            err(f"queries.jsonl:{i}: split은 dev 또는 eval")
        if q.get("modality") == "text" and not str(q.get("text") or "").strip():
            err(f"queries.jsonl:{i}: 텍스트 질의가 비어 있음")
        if q.get("modality") == "image" and not q.get("image_path"):
            err(f"queries.jsonl:{i}: 이미지 질의에 image_path가 없음")

    target_ids = set()
    for i, t in enumerate(targets, 1):
        if t.get("target_id") in target_ids:
            err(f"targets.jsonl:{i}: target_id 중복")
        target_ids.add(t.get("target_id"))
        if t.get("video_id") not in videos:
            err(f"targets.jsonl:{i}: manifest에 없는 영상")

    for i, r in enumerate(qrels, 1):
        if r.get("query_id") not in query_ids:
            err(f"qrels.jsonl:{i}: queries에 없는 query_id")
        if r.get("target_id") not in target_ids:
            err(f"qrels.jsonl:{i}: targets에 없는 target_id")
        if r.get("gt_track_uid") not in uids:
            err(f"qrels.jsonl:{i}: GT에 없는 트랙 {r.get('gt_track_uid')}")
        m = videos.get(r.get("video_id"))
        if m is None or not (type(r.get("start_ms")) is int and type(r.get("end_ms")) is int
                             and 0 <= r["start_ms"] < r["end_ms"] <= m["duration_ms"]):
            err(f"qrels.jsonl:{i}: 영상 또는 구간이 올바르지 않음")

    for i, p in enumerate(predictions, 1):
        if p.get("query_id") not in query_ids:
            err(f"predictions.jsonl:{i}: queries에 없는 query_id")
        ranks = [r.get("rank") for r in p.get("results", [])]
        if ranks != list(range(1, len(ranks) + 1)):
            err(f"predictions.jsonl:{i}: rank가 1부터 연속이 아님")

    if errors:
        raise UserError("평가 입력 검사 실패:\n  " + "\n  ".join(errors))
    return dict(videos=len(manifest), gt_rows=len(gt), gt_tracks=len(uids), queries=len(queries),
                qrels=len(qrels), targets=len(targets), predictions=len(predictions))