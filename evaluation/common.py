"""평가 공통 유틸 (F 5~7). 데이터셋과 무관하게 동작한다 (CityFlow 담당과 공유)."""
from __future__ import annotations

import bisect
import json
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_eval_config(path=None) -> dict:
    with open(path or ROOT / "configs/evaluation/evaluation.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def intersect(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    return None if x2 <= x1 or y2 <= y1 else [x1, y1, x2, y2]


class TimeIndex:
    """트랙(또는 영상)별 관측을 시각으로 찾는 인덱스: 허용차 이내 가장 가까운 관측."""

    def __init__(self, rows, key):
        self.by = defaultdict(list)
        for r in rows:
            self.by[r[key]].append(r)
        self.times = {}
        for k, v in self.by.items():
            v.sort(key=lambda r: r["timestamp_ms"])
            self.times[k] = [r["timestamp_ms"] for r in v]

    def nearest(self, k, t, tol):
        ts = self.times.get(k)
        if not ts:
            return None
        i = bisect.bisect_left(ts, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts) and abs(ts[j] - t) <= tol:
                if best is None or abs(ts[j] - t) < abs(ts[best] - t):
                    best = j
        return None if best is None else self.by[k][best]
