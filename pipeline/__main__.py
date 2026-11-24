"""G 통합 실행기. python -m pipeline {doctor|validate|run|evaluate|serve|experiment|report}"""
import argparse
import contextlib
import csv
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .core import (POLICIES, ROOT, RUNS, Stages, UserError, file_sha256, git_commit, gpu_peak, link_or_copy,
                   load_policy, load_yaml, now, read_json, read_jsonl, rel, within, write_json_atomic, write_jsonl)
from .checks import load_inputs, validate_eval_fixture, validate_observations

RUN_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}")
EMBEDDING_CONFIG = ROOT / "configs/embedding.siglip2.yaml"
DECODER_ID = "pipeline-sequential-pyav-v1"
DEFAULTS = {"kind": "video", "policy": "hybrid", "device": None, "start_ms": None, "end_ms": None,
            "dataset": "meva", "split": "dev", "mode": "pred", "representation": "combined",
            "eval_config": None, "filter_inputs": True}


def settings(args):
    cfg = load_yaml(rel(args.config)) if getattr(args, "config", None) else {}
    out = {**DEFAULTS, **cfg}
    for key in ("input", "policy", "device", "start_ms", "end_ms", "dataset", "split", "mode", "representation"):
        value = getattr(args, key, None)
        if value is not None:
            out[key] = value
    if not out.get("input"):
        raise UserError("--input 또는 설정 파일의 input이 필요합니다.")
    return out


def make_encoder(device):
    from embedding import create_encoder
    from team_search.__main__ import get_config
    config = get_config(SimpleNamespace(embedding_config=EMBEDDING_CONFIG, model_root=None, device=device))
    return config, create_encoder(config)


def register_job(run, key, events, vectors):
    db = sqlite3.connect(run / "events.sqlite3", timeout=10)
    with contextlib.closing(db), db:
        db.execute("""CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
            state TEXT NOT NULL, events INTEGER DEFAULT 0, vectors INTEGER DEFAULT 0,
            error TEXT DEFAULT '')""")
        db.execute("INSERT OR REPLACE INTO jobs(id, kind, name, state, events, vectors, error) "
                   "VALUES(?,?,?,?,?,?,'')", (key, "handoff", key, "ready", events, vectors))


# ---------------------------------------------------------------- doctor
def cmd_doctor(a):
    report = {"time": now(), "python": platform.python_version(), "platform": platform.platform(),
              "cpu_count": os.cpu_count(), "commit": git_commit(), "packages": {}, "problems": [], "warnings": []}
    problems, warnings = report["problems"], report["warnings"]
    if sys.version_info[:2] != (3, 11):
        warnings.append(f"Python 3.11 권장(현재 {platform.python_version()})")
    for dist in ("torch", "torchvision", "av", "numpy", "Pillow", "PyYAML",
                 "fastapi", "uvicorn", "python-multipart", "pydantic"):
        try:
            report["packages"][dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            report["packages"][dist] = None
            problems.append(f"패키지 없음: {dist}")
    try:
        import psutil
        report["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        report["ram_gb"] = None
    try:
        import torch
        report["torch_cuda_build"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": props.name, "vram_gb": round(props.total_memory / 2 ** 30, 1)}
        if a.device == "cuda" and not torch.cuda.is_available():
            problems.append("--device cuda 이지만 CUDA를 쓸 수 없습니다(CPU용 PyTorch이거나 드라이버 문제).")
    except Exception as exc:
        problems.append(f"torch import 실패: {exc}")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        report["nvidia_smi"] = (out.stdout or out.stderr).strip()
    except Exception as exc:
        report["nvidia_smi"] = f"실행 불가: {exc}"
    for module in ("sampling.core", "embedding", "embedding.contracts", "team_search.inputs",
                   "team_search.live", "evaluation.evaluate"):
        try:
            importlib.import_module(module)
        except Exception as exc:
            problems.append(f"모듈 import 실패 {module}: {type(exc).__name__}: {exc}")
    if a.config:
        path = rel(a.config)
        if path.is_file():
            local = load_yaml(path)
            report["local_config"] = local
            for key in ("data_root", "model_root"):
                if key in local and not Path(str(local[key])).exists():
                    problems.append(f"{key} 경로가 없습니다: {local[key]}")
        else:
            warnings.append(f"로컬 설정이 없습니다(configs/local.example.yaml을 복사하세요): {path}")
    free_gb = shutil.disk_usage(ROOT).free // 2 ** 30
    report["disk_free_gb"] = free_gb
    if free_gb < 10:
        warnings.append(f"디스크 여유 {free_gb}GB")
    RUNS.mkdir(exist_ok=True)
    write_json_atomic(RUNS / "env_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if problems else 0


# -------------------------------------------------------------- validate
def cmd_validate(a):
    inp = rel(a.input)
    if not inp.is_dir():
        raise UserError(f"폴더가 없습니다: {inp}")
    if (inp / "tracks_gt.jsonl").is_file():
        summary = validate_eval_fixture(inp)
    else:
        config = load_policy(a.policy)
        data = load_inputs(inp, a.start_ms, a.end_ms)
        media = not a.no_media
        summary = validate_observations(config, data, inp, check_media=media, check_hash=media,
                                        require_split=media, windowed=a.start_ms is not None or a.end_ms is not None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------- run
def execute_run(a, st, run, meta, stages):
    from sampling.core import run_sampling
    fixture = st["kind"] == "fixture"
    config = load_policy(st["policy"])
    write_json_atomic(run / "configs" / f"{st['policy']}.json", config)
    inp = rel(st["input"])
    windowed = st["start_ms"] is not None or st["end_ms"] is not None

    with stages.stage("validate"):
        data = load_inputs(inp, st["start_ms"], st["end_ms"])
        summary = validate_observations(config, data, inp, check_media=not fixture, check_hash=False,
                                        require_split=not fixture, windowed=windowed)
        for warning in summary["warnings"]:
            print("[경고]", warning)
        meta["input_sha256"] = {n: file_sha256(inp / n) for n in (
            "manifest.jsonl", "frames.jsonl", "tracks.jsonl", "video_status.jsonl") if (inp / n).is_file()}
        meta["videos"] = [m["video_id"] for m in data[0]]
    manifests, frames, tracks, statuses, exclusions = data

    loader, motion_reader = None, None
    with stages.stage("sampling"):
        if config["sampling"]["policy"] == "motion":
            if fixture:
                raise UserError("motion 정책은 영상 파일이 필요합니다.")
            from .reader import SequentialReader
            motion_reader = SequentialReader({
                m["video_id"]: (within(inp, m["relative_path"]), int(m.get("source_offset_ms") or 0), m.get("time_source"))
                for m in manifests})

            def loader(frame):
                return motion_reader.read_frame(frame["video_id"], frame["timestamp_ms"])["image_rgb"]
        try:
            events, samples, selection = run_sampling(config, manifests, frames, tracks, statuses, exclusions,
                                                      a.run_id, frame_loader=loader)
        finally:
            if motion_reader is not None:
                motion_reader.close()
        motion_scores = selection.pop("motion_scores", None)
        write_jsonl(run / "events.jsonl", events)
        write_jsonl(run / "samples.jsonl", samples)
        if motion_scores is not None:
            write_jsonl(run / "motion_scores.jsonl", motion_scores)
        write_json_atomic(run / "selection_summary.json", selection)
        targets = config.get("target_classes")
        tracks_out = [t for t in tracks if not targets or t["class_name"] in targets]
        write_jsonl(run / "frames.jsonl", frames)
        write_jsonl(run / "tracks.jsonl", tracks_out)
        want = (st.get("expect") or {}).get("counts")
        if want:
            got = selection.get("counts") or {}
            bad = {k: {"실측": got.get(k), "기대": v} for k, v in want.items() if got.get(k) != v}
            if bad:
                raise UserError(f"A 예제 기대값 불일치: {bad}")
            print(f"[PASS] A 예제 기대값과 일치 {want}")
        if not samples:
            raise UserError("이 구간에서 검색에 사용할 장면이 선택되지 않았습니다.")
        print(f"[샘플링] event {len(events)}건, sample {len(samples)}건")

    info = {"events": len(events), "samples": len(samples)}
    if fixture:
        with stages.stage("predict"):
            source = rel(st["predictions_fixture"])
            if not source.is_file():
                raise UserError(f"예측 예제가 없습니다: {source}")
            shutil.copyfile(source, run / "predictions.jsonl")
        return info

    folder = run / "indexes" / a.run_id
    index_dir = folder / "index"
    specs = {}
    with stages.stage("media"):
        index_rows = []
        for m in manifests:
            src = within(inp, m["relative_path"])
            dst = within(folder / "input", m["relative_path"])
            link_or_copy(src, dst)
            specs[m["video_id"]] = (dst, int(m.get("source_offset_ms") or 0), m.get("time_source"))
            index_rows.append(dict(m, display_name=m.get("display_name") or m["video_id"]))
        write_jsonl(folder / "manifest.jsonl", index_rows)
        print(f"[영상] 등록 {len(specs)}개: {sorted(specs)}")

    with stages.stage("model_load"):
        embedding_config, encoder = make_encoder(st["device"])
        meta["device_actual"] = str(getattr(encoder, "device", None))
        meta["encoder"] = getattr(encoder, "metadata", None)

    with stages.stage("embedding"):
        from embedding import build_index
        from .reader import SequentialReader
        missing = sorted({s["video_id"] for s in samples} - set(specs))
        if missing:
            raise UserError(f"샘플의 video_id가 영상 목록에 없습니다: {missing} / 영상 목록: {sorted(specs)}")
        reader = SequentialReader(specs)
        batch_key = "batch_size_gpu" if encoder.device.startswith("cuda") else "batch_size_cpu"
        try:
            status = build_index(run / "samples.jsonl", folder / "manifest.jsonl", folder / "input", index_dir,
                                 ROOT / ".cache/embedding", reader.read_frame, encoder, DECODER_ID,
                                 embedding_config["crop_margin_ratio"], embedding_config[batch_key])
        finally:
            reader.close()
        if status.get("status") != "success":
            errors_path = index_dir / "errors.jsonl"
            first = read_jsonl(errors_path)[:3] if errors_path.is_file() else []
            raise ValueError("임베딩 실패: " + json.dumps(first, ensure_ascii=False))

    with stages.stage("register"):
        index = json.loads((index_dir / "index_manifest.json").read_text(encoding="utf-8"))
        info["vectors"] = index["row_count"]
        (folder / "counts.json").write_text(json.dumps([len(events), info["vectors"]]), encoding="utf-8")
        register_job(run, a.run_id, len(events), info["vectors"])
        for name in ("items.jsonl", "vectors.npy", "index_manifest.json", "errors.jsonl", "embedding_status.json"):
            if (index_dir / name).is_file():
                link_or_copy(index_dir / name, run / name)

    with stages.stage("predict"):
        from .search import eval_queries, predict
        queries = eval_queries(st["dataset"], st["split"], set(specs))
        if not queries:
            print("[경고] 이 run의 영상에 정답이 있는 텍스트 질의가 없어 predictions를 만들지 않습니다.")
        else:
            from rag.engine import FrameRecommender
            engine = FrameRecommender(index_dir, encoder)
            rows, info["search"] = predict(engine, encoder, queries, read_jsonl(run / "tracks.jsonl"),
                                           st["representation"])
            write_jsonl(run / "predictions.jsonl", rows)
            print(f"[검색] 질의 {len(rows)}개, p50 {info['search']['p50_ms']:.0f}ms")
    return info


def cmd_run(a):
    st = settings(a)
    if not RUN_ID.fullmatch(a.run_id):
        raise UserError("run_id는 영문·숫자·_·-만 쓸 수 있고 80자 이하입니다.")
    run = RUNS / a.run_id
    if run.exists() and any(run.iterdir()):
        raise UserError(f"이미 있는 run_id입니다. 새 이름을 쓰세요: {a.run_id}")
    (run / "configs").mkdir(parents=True, exist_ok=True)
    stages = Stages(run)
    rep = st["representation"]
    meta = {"run_id": a.run_id, "status": "running", "started": now(), "commit": git_commit(),
            "experiment_id": getattr(a, "experiment_id", None), "kind": st["kind"], "policy": st["policy"],
            "device": st["device"], "input": str(rel(st["input"])), "window_ms": [st["start_ms"], st["end_ms"]],
            "dataset": st["dataset"], "split": st["split"], "mode": st["mode"], "representation": rep,
            "representations": st.get("representations") or (["object", "scene"] if rep == "combined" else [rep]),
            "eval_config": st.get("eval_config"), "filter_inputs": bool(st.get("filter_inputs", True)),
            "expect": st.get("expect")}
    write_json_atomic(run / "run_manifest.json", meta)
    try:
        info = execute_run(a, st, run, meta, stages)
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished=now())
        write_json_atomic(run / "run_manifest.json", meta)
        raise
    index_dir = run / "indexes" / a.run_id / "index"
    size = sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file()) if index_dir.is_dir() else 0
    write_json_atomic(run / "costs.json", {
        "stages_ms": {k: v.get("wall_ms") for k, v in stages.data.items()}, "cache": "cold 또는 .cache/embedding 재사용(구분 기록 없음)",
        "events": info.get("events"), "samples": info.get("samples"), "vectors": info.get("vectors"),
        "index_bytes": size, "gpu_peak_allocated_bytes": gpu_peak(), "search": info.get("search")})
    meta.update(status="success", finished=now())
    write_json_atomic(run / "run_manifest.json", meta)
    print(f"[완료] {run}\n평가: python -m pipeline evaluate --run-id {a.run_id}")
    return 0


def run_reuse(ns, base_id):
    """E05·E06처럼 다른 run의 samples·인덱스를 재사용하고 검색 표현만 바꾼다."""
    from .search import eval_queries, predict
    st = settings(ns)
    base = RUNS / base_id
    base_meta = read_json(base / "run_manifest.json")
    if base_meta.get("status") != "success":
        raise UserError(f"재사용할 run이 성공 상태가 아닙니다: {base_id}")
    run = RUNS / ns.run_id
    if run.exists() and any(run.iterdir()):
        raise UserError(f"이미 있는 run_id입니다: {ns.run_id}")
    run.mkdir(parents=True, exist_ok=True)
    stages = Stages(run)
    rep = st["representation"]
    meta = dict(base_meta, run_id=ns.run_id, status="running", started=now(), commit=git_commit(),
                reused_from=base_id, experiment_id=getattr(ns, "experiment_id", None),
                representation=rep, representations=[rep])
    write_json_atomic(run / "run_manifest.json", meta)
    try:
        for name in ("samples.jsonl", "items.jsonl", "events.jsonl", "frames.jsonl", "tracks.jsonl", "index_manifest.json"):
            if (base / name).is_file():
                link_or_copy(base / name, run / name)
        with stages.stage("model_load"):
            _, encoder = make_encoder(st["device"])
        search = None
        with stages.stage("predict"):
            from rag.engine import FrameRecommender
            queries = eval_queries(meta["dataset"], meta["split"], set(meta.get("videos") or []))
            if queries:
                engine = FrameRecommender(base / "indexes" / base_id / "index", encoder)
                rows, search = predict(engine, encoder, queries, read_jsonl(run / "tracks.jsonl"), rep)
                write_jsonl(run / "predictions.jsonl", rows)
        write_json_atomic(run / "costs.json", {"reused_from": base_id, "search": search,
                                               "stages_ms": {k: v.get("wall_ms") for k, v in stages.data.items()}})
    except BaseException as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished=now())
        write_json_atomic(run / "run_manifest.json", meta)
        raise
    meta.update(status="success", finished=now())
    write_json_atomic(run / "run_manifest.json", meta)
    return 0


# -------------------------------------------------------------- evaluate
def build_run_eval_config(run, base_path, dataset, split, video_ids):
    import yaml
    cfg = load_yaml(base_path)
    ds = cfg["datasets"][dataset]
    ids = set(video_ids)
    manifest = [m for m in read_jsonl(rel(ds["manifest"])) if m.get("video_id") in ids]
    if not manifest:
        raise UserError(f"F manifest({ds['manifest']})에 이 run의 video_id가 없습니다: {sorted(ids)}")
    found = {m["video_id"] for m in manifest}
    qrels = [r for r in read_jsonl(rel(ds["qrels"])) if r.get("video_id") in found]
    qids = {r["query_id"] for r in qrels}
    queries = [q for q in read_jsonl(rel(ds["queries"])) if q.get("query_id") in qids]
    targets = [t for t in read_jsonl(rel(ds["targets"])) if t.get("video_id") in found]
    folder = run / "eval_inputs"
    for name, rows in (("manifest", manifest), ("qrels", qrels), ("queries", queries), ("targets", targets)):
        write_jsonl(folder / f"{name}.jsonl", rows)
    cfg["datasets"][dataset] = {**ds, **{k: (folder / f"{k}.jsonl").as_posix()
                                         for k in ("manifest", "qrels", "queries", "targets")}}
    path = run / "configs" / "evaluation.run.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def cmd_evaluate(a):
    run = RUNS / a.run_id
    meta = read_json(run / "run_manifest.json")
    if not meta:
        raise UserError(f"run이 없습니다: {run}")
    if meta.get("status") != "success":
        raise UserError(f"성공한 run만 평가합니다(현재 {meta.get('status')}).")
    dataset = a.dataset or meta.get("dataset") or "meva"
    split = a.split or meta.get("split") or "dev"
    mode = a.mode or meta.get("mode") or "pred"
    reps = a.representations or meta.get("representations") or ["object", "scene"]
    explicit = a.eval_config or meta.get("eval_config")
    config_path = rel(explicit) if explicit else ROOT / "configs/evaluation.yaml"
    if a.eval_config is None and meta.get("filter_inputs", True):
        config_path = build_run_eval_config(run, config_path, dataset, split, meta.get("videos") or [])
    ds = load_yaml(config_path)["datasets"][dataset]
    gt = rel(ds["gt_tracks"])
    if not gt.is_file():
        raise UserError(f"정답 트랙 파일이 없습니다: {gt}\n  F가 변환한 tracks_gt.jsonl이 필요합니다(Git에 포함되지 않음).")
    if not (run / "predictions.jsonl").is_file():
        raise UserError("predictions.jsonl이 없습니다(run에서 질의가 없었거나 predict 단계를 건너뜀).")
    module = importlib.import_module("evaluation.evaluate")
    code = module.main(["--dataset", dataset, "--split", split, "--run-dir", str(run), "--mode", mode,
                        "--representations", *reps, "--config", str(config_path)])
    metrics = read_json(run / "metrics.json")
    allm = (metrics.get("retrieval") or {}).get("all") or {}
    if allm:
        print(f"R@5={allm['recall@5']:.4f} R@10={allm['recall@10']:.4f} MRR={allm['mrr']:.4f} "
              f"(질의 {allm['n_queries']}개, 예측 누락 {allm.get('n_missing_prediction')})")
    else:
        print(f"[참고] 검색 지표가 없습니다: status={metrics.get('status')} reason={metrics.get('reason')}")
    want = (meta.get("expect") or {}).get("metrics")
    if want:
        bad = {k: {"실측": allm.get(k), "기대": v} for k, v in want.items()
               if allm.get(k) is None or abs(allm[k] - v) > 1e-3}
        if bad:
            raise UserError(f"기대 지표와 다릅니다: {bad}")
        print("[PASS] 기대 지표와 일치")
    meta["evaluation"] = {"dataset": dataset, "split": split, "mode": mode, "config": str(config_path), "at": now()}
    write_json_atomic(run / "run_manifest.json", meta)
    return code


# ----------------------------------------------------------------- serve
def cmd_serve(a):
    if a.host != "127.0.0.1":
        raise UserError("서비스는 127.0.0.1에서만 실행합니다.")
    run = RUNS / a.run_id
    if not (run / "indexes" / a.run_id / "index" / "index_manifest.json").is_file():
        raise UserError(f"웹으로 띄울 인덱스가 있는 run이 아닙니다: {run}")
    device = a.device or read_json(run / "run_manifest.json").get("device")
    command = [sys.executable, "-X", "utf8", "-m", "team_search.live", "--store", str(run),
               "--port", str(a.port), "--llm-port", str(a.llm_port)]
    if device in ("cpu", "cuda", "auto"):
        command += ["--device", device]
    print(f"http://127.0.0.1:{a.port}/ui/   (종료: Ctrl+C)")
    try:
        return subprocess.call(command, cwd=ROOT)
    except KeyboardInterrupt:
        return 0


# ------------------------------------------------------------ experiment
def cmd_experiment(a):
    matrix = load_yaml(rel(a.matrix))
    common, experiments = matrix.get("common") or {}, matrix.get("experiments") or []
    if not experiments:
        raise UserError("experiments가 비어 있습니다.")
    prefix = a.prefix or matrix.get("prefix") or datetime.now().strftime("x%m%d%H%M")
    if not RUN_ID.fullmatch(prefix):
        raise UserError("prefix는 영문·숫자·_·-만 쓸 수 있습니다.")
    results = []
    for exp in experiments:
        merged = {**common, **{k: v for k, v in exp.items() if k not in ("id", "reuse")}}
        ns = SimpleNamespace(config=None, run_id=f"{prefix}-{exp['id']}", experiment_id=exp["id"],
                             input=merged.get("input"), policy=merged.get("policy"),
                             device=a.device or merged.get("device"), start_ms=merged.get("start_ms"),
                             end_ms=merged.get("end_ms"), dataset=merged.get("dataset"), split=merged.get("split"),
                             mode=merged.get("mode"), representation=merged.get("representation"))
        try:
            code = run_reuse(ns, f"{prefix}-{exp['reuse']}") if exp.get("reuse") else cmd_run(ns)
            status = "success" if code == 0 else "failed"
        except Exception as exc:  # 한 실험의 실패가 나머지 실험을 막지 않는다
            status = "failed"
            print(f"[실험 {exp['id']} 실패] {type(exc).__name__}: {exc}", file=sys.stderr)
        results.append({"id": exp["id"], "run_id": ns.run_id, "status": status})
    write_json_atomic(RUNS / f"{prefix}_experiment.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"다음: python -m pipeline report --runs runs --prefix {prefix} --out reports/{prefix}")
    return 0 if all(r["status"] == "success" for r in results) else 1


# ---------------------------------------------------------------- report
def cmd_report(a):
    runs = rel(a.runs)
    if not runs.is_dir():
        raise UserError(f"runs 폴더가 없습니다: {runs}")
    out = rel(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, statuses = [], []
    for path in sorted(p for p in runs.iterdir() if p.is_dir()):
        meta = read_json(path / "run_manifest.json")
        if not meta or (a.prefix and not path.name.startswith(a.prefix)):
            continue
        statuses.append((path.name, meta.get("status"), meta.get("error") or ""))
        if meta.get("status") != "success":
            continue
        metrics, costs = read_json(path / "metrics.json"), read_json(path / "costs.json")
        allm = (metrics.get("retrieval") or {}).get("all") or {}
        track = (metrics.get("retention") or {}).get("track") or {}
        stages = costs.get("stages_ms") or {}
        search = costs.get("search") or {}
        rows.append({
            "run_id": path.name, "experiment": meta.get("experiment_id"), "policy": meta.get("policy"),
            "representation": meta.get("representation"), "dataset": meta.get("dataset"), "split": meta.get("split"),
            "mode": meta.get("mode"), "reused_from": meta.get("reused_from"), "evaluated": bool(allm),
            "queries": allm.get("n_queries"), "candidates": search.get("candidates"),
            "recall@5": allm.get("recall@5"), "recall@10": allm.get("recall@10"), "mrr": allm.get("mrr"),
            "track_retention": track.get("retention_rate"), "track_missing_rate": track.get("missing_rate"),
            "index_ms": sum(v for k, v in stages.items() if k in ("sampling", "media", "embedding") and v),
            "vectors": costs.get("vectors"), "index_bytes": costs.get("index_bytes"),
            "gpu_peak_bytes": costs.get("gpu_peak_allocated_bytes"),
            "search_p50_ms": search.get("p50_ms"), "search_p95_ms": search.get("p95_ms"),
            "search_samples": search.get("n_queries")})
    columns = list(rows[0]) if rows else ["run_id"]
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def cell(v):
        return "" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
    lines = ["# 실행 결과 요약", "", f"생성: {now()}", "", "## 성공한 실행", "",
             "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(r[c]) for c in columns) + " |" for r in rows]
    lines += ["", "## 상태 목록(실패·누락 포함)", "", "| run_id | status | error |", "|---|---|---|"]
    lines += [f"| {r} | {s} | {e} |" for r, s, e in statuses]
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"성공 {len(rows)}건 / 전체 {len(statuses)}건 -> {out}")
    return 0


# ------------------------------------------------------------------ main
def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pipeline", description="G 통합 실행기")
    sub = parser.add_subparsers(dest="command", required=True)
    device = ("cpu", "cuda", "auto")

    d = sub.add_parser("doctor", help="환경 점검")
    d.add_argument("--config", help="configs/local.yaml")
    d.add_argument("--device", choices=device)

    v = sub.add_parser("validate", help="B 산출물 또는 F 평가 입력 검사")
    v.add_argument("--input", "--fixture", dest="input", required=True)
    v.add_argument("--policy", choices=sorted(POLICIES), default="hybrid")
    v.add_argument("--start-ms", type=int)
    v.add_argument("--end-ms", type=int)
    v.add_argument("--no-media", action="store_true", help="영상 파일 존재·해시 검사 생략(예제용)")

    r = sub.add_parser("run", help="검사→샘플링→임베딩→검색 예측")
    r.add_argument("--config")
    r.add_argument("--input")
    r.add_argument("--run-id", required=True)
    r.add_argument("--policy", choices=sorted(POLICIES))
    r.add_argument("--start-ms", type=int)
    r.add_argument("--end-ms", type=int)
    r.add_argument("--device", choices=device)
    r.add_argument("--dataset")
    r.add_argument("--split", choices=("dev", "eval"))
    r.add_argument("--mode", choices=("gt", "pred"))
    r.add_argument("--representation", choices=("combined", "scene", "object"))

    e = sub.add_parser("evaluate", help="F의 평가기 호출")
    e.add_argument("--run-id", required=True)
    e.add_argument("--dataset")
    e.add_argument("--split", choices=("dev", "eval"))
    e.add_argument("--mode", choices=("gt", "pred"))
    e.add_argument("--representations", nargs="*")
    e.add_argument("--eval-config")

    s = sub.add_parser("serve", help="run의 인덱스를 웹 화면으로 띄움")
    s.add_argument("--run-id", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8780)
    s.add_argument("--llm-port", type=int, default=8781)
    s.add_argument("--device", choices=device)

    x = sub.add_parser("experiment", help="E01~E08 실험 실행")
    x.add_argument("--matrix", required=True)
    x.add_argument("--prefix")
    x.add_argument("--device", choices=device)

    p = sub.add_parser("report", help="결과·비용 집계")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", required=True)
    p.add_argument("--prefix")

    args = parser.parse_args(argv)
    commands = {"doctor": cmd_doctor, "validate": cmd_validate, "run": cmd_run, "evaluate": cmd_evaluate,
                "serve": cmd_serve, "experiment": cmd_experiment, "report": cmd_report}
    try:
        return commands[args.command](args)
    except (UserError, ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        print(f"[입력·검사 오류] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        traceback.print_exc()
        print(f"[내부 오류] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())