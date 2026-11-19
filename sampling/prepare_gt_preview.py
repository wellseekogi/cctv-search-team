"""Build a dev-only C handoff without inventing decoder or completion evidence."""
import argparse
import copy
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from .__main__ import read_jsonl, write_json, write_jsonl
from .core import run_sampling, require


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    require(not args.out.exists(), "Choose a new output folder")
    manifests = read_jsonl(args.source / "manifest.jsonl")
    all_tracks = read_jsonl(args.source / "tracks_gt.jsonl")
    manifest = [dict(m, schema_version="cctv-3.0") for m in manifests if m["split"] == "dev" and m["camera_id"] == "S05_c033"]
    require(len(manifest) == 1, "Expected exactly one dev c033 video")
    m = manifest[0]
    dev = [o for o in all_tracks if o["video_id"] == m["video_id"]]
    require(all(o["timestamp_ms"] == o["frame_index"] * 100 for o in dev), "Expected supplied 10Hz annotations")
    # Use the project's 200ms grid, without claiming these frames were decoded.
    tracks = [o for o in dev if o["timestamp_ms"] % 200 == 0]
    frames = [dict(schema_version="cctv-3.0", video_id=m["video_id"], frame_index=t//100,
                   timestamp_ms=t, requested_timestamp_ms=t, decode_status="unverified", detector_status="skipped",
                   time_source="gt_annotation_unverified", width=m["width"], height=m["height"])
              for t in sorted({o["timestamp_ms"] for o in tracks})]
    statuses = [dict(video_id=m["video_id"], video_status="unverified", available_end_ms=m["available_end_ms"])]
    config = json.loads((root / "configs/sampling.yaml").read_text(encoding="utf-8"))
    config["execution_mode"] = "annotation_preview"
    config["preview_contract"] = dict(candidate_step_ms=200, split="dev", camera="c033",
                                      query_exclusion_status="not_provided", threshold_tuning=False)
    events, samples, summary = run_sampling(config, manifest, frames, tracks, statuses, [], "c033-gt-preview-v1")
    again = run_sampling(config, manifest, list(reversed(frames)), list(reversed(tracks)), statuses, [], "c033-gt-preview-v1")
    require((events, samples, summary) == again, "Reproducibility failed")
    ids = {s["item_id"]: s for s in samples}
    require(len(ids) == len(samples), "Duplicate sample IDs")
    for ev in events:
        for kind in ("object", "scene"):
            item = ev[kind + "_item_id"]
            require(item is None or (item in ids and ev["event_id"] in ids[item]["event_ids"]), "Broken event/sample link")
    counts = Counter(s["track_uid"] for s in samples if s["representation"] == "object")
    require(all(n <= 30 for n in counts.values()), "Object cap exceeded")
    require(all(s["video_id"] == m["video_id"] for s in samples), "Eval leakage")
    summary.update(input_observations=len(dev), grid_observations=len(tracks), source_track_count=len({o["track_uid"] for o in dev}),
                   source_sha256={name: hashlib.sha256((args.source/name).read_bytes()).hexdigest()
                                  for name in ("manifest.jsonl", "tracks_gt.jsonl")})
    args.out.mkdir(parents=True)
    inputs = args.out / "input_dev"
    inputs.mkdir()
    for name, rows in (("manifest.jsonl", manifest), ("tracks.jsonl", tracks), ("frames.jsonl", frames),
                       ("video_status.jsonl", statuses), ("query_exclusions.jsonl", [])):
        write_jsonl(inputs / name, rows)
    write_jsonl(args.out / "manifest.jsonl", manifest)
    write_jsonl(args.out / "events.jsonl", events)
    write_jsonl(args.out / "samples.jsonl", samples)
    # Small C smoke example: a matched scene/object pair from one actual annotation frame.
    scene_keys = {(s["video_id"], s["frame_index"]) for s in samples if s["representation"] == "scene"}
    obj = next(s for s in samples if s["representation"] == "object" and (s["video_id"], s["frame_index"]) in scene_keys)
    smoke = [s for s in samples if s["item_id"] == obj["item_id"] or
             (s["representation"] == "scene" and s["frame_index"] == obj["frame_index"])]
    write_jsonl(args.out / "samples_smoke.jsonl", smoke)
    write_json(args.out / "sampling.yaml", config)
    write_json(args.out / "selection_summary.json", summary)
    write_json(args.out / "verification.json", {"status": "PASS", "verifier": "automated assertions",
               "checks": ["input schema/time/bbox validation", "two runs identical with reversed input order",
                          "unique sample IDs", "event/sample links", "object cap", "dev c033 only"],
               "counts": summary["counts"], "C_acceptance": "pending", "media_verified": False})
    write_jsonl(args.out / "errors.jsonl", [])
    code = args.out / "sampling"
    code.mkdir()
    for name in ("__init__.py", "core.py", "uniform.py", "__main__.py", "prepare_gt_preview.py"):
        (code/name).write_bytes((root/"sampling"/name).read_bytes())
    readme = f'''# C 전달용 dev c033 GT 임시 선택 결과

실제 제공 GT 주석으로 만든 연결용 예제입니다. 최종 평가용 결과가 아닙니다.
원본 영상은 포함하지 않았습니다. 먼저 samples_smoke.jsonl의 scene/object 2행으로 연결을 확인해주세요.

- 전체 선택: object {summary['counts']['object']}개, scene {summary['counts']['scene']}개
- Event {summary['counts']['events']}건: TRACK_START, POSITION_SHIFT만 사용
- dev c033만 사용. eval c007은 미실행. 임계값은 명세 기본값 그대로이며 튜닝하지 않았습니다.
- 원본 100ms GT 관측 중 200ms 격자 관측만 사용했습니다.
- blur_var가 없어서 QUALITY_GAIN 미판정.
- B의 실제 frames 로그가 없어서 TRACK_GAP/RESUME 미판정. 해당 트랙의 관측 공백을 해석하지 않습니다.
- 정상 영상 완료 증거가 없어서 VIDEO_END 미판정.
- F의 이미지 질의 제외 구간 미수신: 제외 규칙 미적용. 최종 인덱스/성능평가에는 사용하지 마세요.
- 시각/프레임 번호는 GT 주석 기준이고 실제 디코딩 시각을 검증하지 않았습니다.
- input_dev/frames.jsonl은 GT에서 유도한 임시 목록입니다. B가 만든 로그가 아니며 decode_status=unverified입니다.
- 최종 재실행에서는 설정/ID/선택 개수가 달라질 수 있습니다. 캐시를 이 preview 실행과 분리해주세요.

## C가 할 일

1. manifest.jsonl의 relative_path를 본인의 data_root에 연결합니다.
2. samples_smoke.jsonl의 두 행으로 전체 장면과 객체 crop을 생성해봅니다.
3. frame_index는 0 기반, bbox는 원본 해상도의 [x1,y1,x2,y2]입니다.
4. scene의 bbox/track_uid는 null입니다. object crop 여백은 C가 명세대로 적용합니다.
5. 실제 RGB 읽기에는 원본 영상과 B의 read_frame 함수가 필요합니다.
6. 실제 반환 시각·박스가 맞는지 확인하고 PASS/FAIL을 A에게 알려주세요.

## A 결과 재현

압축을 푼 폴더에서 다음 명령을 실행합니다. Python 표준 라이브러리만 사용합니다.

    python -m sampling --input input_dev --config sampling.yaml --out reproduced --run-id c033-gt-preview-v1

sampling.yaml은 JSON 문법의 YAML입니다. 이미 결과가 있는 out 폴더는 덮어쓰지 않습니다.
selection_summary의 원본 파일 해시는 A가 받은 원본 전체 파일 기준입니다.
재현 명령은 normalized input_dev의 해시를 기록하므로 요약 파일의 부가 필드는 다를 수 있습니다.
events와 samples는 동일해야 합니다.

## 다음에 필요한 것

B: 실제 frames.jsonl, read_frame, 품질값, 영상 완료 상태.
F: 이미지 질의 제거 구간(없으면 명시적 확인), GT 시간/박스의 원본 대조.
C: 원본 영상 확보 및 위 smoke 확인 결과.
'''
    (args.out/"C_README.md").write_text(readme, encoding="utf-8")
    archive = args.out.with_suffix(".zip")
    require(not archive.exists(), "Archive already exists")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.out.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(args.out))
    print(json.dumps({"counts": summary["counts"], "events": summary["event_types"], "zip": str(archive)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
