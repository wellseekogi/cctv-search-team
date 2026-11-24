"""G 통합 실행기 공통 함수 (경로·JSONL·상태 기록·해시)."""
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUNS = ROOT / "runs"

# 정책 파일은 확장자가 yaml이지만 내용은 JSON이다(A의 CLI도 json.loads로 읽는다).
# PyYAML은 1e-09 같은 값을 문자열로 읽으므로 이 파일들은 json으로 읽는다.
POLICIES = {
    "hybrid": "configs/person/hybrid.yaml",
    "object_event": "configs/person/sampling.yaml",
    "motion": "configs/person/motion.yaml",
    "uniform_1000": "configs/person/uniform_1000.yaml",
    "uniform_2000": "configs/person/uniform_2000.yaml",
    "uniform_5000": "configs/person/uniform_5000.yaml",
    "generic_object_event": "configs/sampling.yaml",
}


class UserError(ValueError):
    """입력·검사 오류 (종료 코드 2)."""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def within(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise UserError(f"경로가 기준 폴더 밖을 가리킵니다: {relative}")
    return path


def file_sha256(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_jsonl(path):
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise UserError(f"{path.name}:{number}: JSON 오류 ({exc})") from exc
            if not isinstance(row, dict):
                raise UserError(f"{path.name}:{number}: 객체가 아닙니다")
            rows.append(row)
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_yaml(path):
    import yaml
    path = Path(path)
    if not path.is_file():
        raise UserError(f"설정 파일이 없습니다: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_policy(name):
    if name not in POLICIES:
        raise UserError(f"알 수 없는 정책: {name} (선택지: {sorted(POLICIES)})")
    return json.loads((ROOT / POLICIES[name]).read_text(encoding="utf-8-sig"))


def link_or_copy(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def git_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def gpu_peak():
    try:
        import torch
        return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    except Exception:
        return None


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class Stages:
    def __init__(self, run):
        self.path, self.data = Path(run) / "stage_status.json", {}

    def save(self):
        write_json_atomic(self.path, self.data)

    @contextlib.contextmanager
    def stage(self, name):
        started = time.perf_counter()
        self.data[name] = {"status": "running"}
        self.save()
        try:
            yield
        except BaseException as exc:
            self.data[name] = {"status": "failed", "wall_ms": round((time.perf_counter() - started) * 1000),
                               "error": f"{type(exc).__name__}: {exc}"}
            self.save()
            raise
        self.data[name] = {"status": "success", "wall_ms": round((time.perf_counter() - started) * 1000)}
        self.save()