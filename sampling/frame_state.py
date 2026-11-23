"""Select scene-state windows directly from decoded pixels, without B observations."""

from bisect import bisect_left
from dataclasses import dataclass, replace
import math
from pathlib import Path

import av
import numpy as np


@dataclass(frozen=True)
class StateWindow:
    start_sec: float
    end_sec: float
    frame_indices: tuple[int, int, int]
    frame_times_sec: tuple[float, float, float]
    roi_xyxy: tuple[float, float, float, float]
    score: float
    source: str


def _tile_change(current, reference):
    delta = current - reference
    # A camera-wide exposure offset should not look like activity in every tile.
    residual = np.abs(delta - np.median(delta))
    return residual.reshape(6, 16, 8, 16).mean(axis=(1, 3))


def _strength(tiles):
    top_two = np.partition(tiles.ravel(), -2)[-2:]
    return float((0.75 * top_two.mean() + 0.25 * tiles.mean()) / 255)


def _scan(path, start, end, time_source):
    observations, times = [], []
    previous = background = previous_deviation = None
    next_sample = start
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = float(stream.average_rate or 0)
        # MEVA's CFR AVI manifests and actor annotations use frame number / FPS.
        # The AVI demuxer assigns some B frames nonmonotone synthetic PTS.
        avi_cfr = time_source == "fps_formula_avi_cfr"
        if avi_cfr and rate <= 0:
            raise ValueError("CFR timestamp calculation requires a positive frame rate")
        first_pts = None
        for index, frame in enumerate(container.decode(stream)):
            if avi_cfr:
                stamp = index / rate
            elif frame.time is not None:
                if first_pts is None:
                    first_pts = frame.time
                stamp = frame.time - first_pts
            elif rate > 0:
                stamp = index / rate
            else:
                raise ValueError("frame has neither timestamp nor frame rate")
            if times and stamp < times[-1]:
                raise ValueError("out-of-order video frame timestamps")
            if stamp >= end:
                break
            times.append(stamp)
            if stamp < next_sample:
                continue
            next_sample = start + math.floor(stamp - start) + 1
            gray = frame.reformat(width=128, height=96, format="gray").to_ndarray().astype(np.float32)
            if previous is None:
                previous = background = gray
                previous_deviation = np.zeros((6, 8), dtype=np.float32)
            changes = _tile_change(gray, previous)
            deviation = _tile_change(gray, background)
            # Require two successive deviating samples before prioritizing a static change.
            sustained = np.minimum(previous_deviation, deviation)
            strength = _strength(changes)
            persistence = 0.6 * _strength(sustained)
            source, tiles, score = ("persistent", sustained, persistence) if persistence > strength else (
                "change", changes, strength
            )
            row, col = np.unravel_index(tiles.argmax(), tiles.shape)
            roi = (max(0, col - 1) / 8, max(0, row - 1) / 6,
                   min(8, col + 2) / 8, min(6, row + 2) / 6)
            observations.append((stamp, score, source, roi))
            background = 0.95 * background + 0.05 * gray
            previous, previous_deviation = gray, deviation
        if not observations:
            raise ValueError("no decoded frames in the requested time range")
        # At a segment boundary, use the next real PTS. At EOF, extend the last
        # real PTS by one frame duration, never to the requested but absent end.
        if stamp < end:
            step = float(frame.duration * frame.time_base) if frame.duration and frame.time_base else 0
            if step <= 0:
                step = 1 / rate if rate > 0 else times[-1] - times[-2] if len(times) > 1 else 0
            if step <= 0:
                raise ValueError("duration of final video frame is unavailable")
            end = min(end, stamp + step)
    return times, observations, end


def _pixel_windows(times, observations, start, end, budget, excluded):
    start_ms, end_ms = round(start * 1000), round(end * 1000)
    by_tick = {math.floor(stamp - start): (score, source, roi) for stamp, score, source, roi in observations}
    windows, centers = [], []
    for tick, center in enumerate(range(start_ms + 2000, end_ms - 1999, 1000), start=2):
        left, right = (center - 2000) / 1000, (center + 2000) / 1000
        if tick not in by_tick or any(a < right and left < b for a, b in excluded):
            continue
        indices = tuple(bisect_left(times, (center + offset) / 1000) for offset in (-1000, 0, 1000))
        if len(set(indices)) < 3 or indices[-1] == len(times) or times[indices[-1]] >= right:
            continue
        score, source, (x1, y1, x2, y2) = by_tick[tick]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # A 1.5x zoom keeps the selected area's center and crops its middle two thirds.
        dx, dy = (x2 - x1) / 3, (y2 - y1) / 3
        roi = (float(cx - dx), float(cy - dy), float(cx + dx), float(cy + dy))
        windows.append(StateWindow(left, right, indices, tuple(times[i] for i in indices), roi, score, source))
        centers.append(center)

    if not windows:
        return []
    centers = np.asarray(centers)
    distances = np.full(len(windows), np.inf)
    chosen = {}

    def choose(index, source):
        chosen[index] = replace(windows[index], source=source)
        np.minimum(distances, np.abs(centers - centers[index]), out=distances)

    duration = end_ms - start_ms
    reserved = min(budget // 3, math.ceil(duration / 20000))
    for slot in range(reserved):
        eligible = np.flatnonzero(distances >= 4000)
        if not len(eligible):
            break
        target = start_ms + (slot + 0.5) * duration / reserved
        choose(eligible[np.argmin(np.abs(centers[eligible] - target))], "keepalive")

    for index in sorted(range(len(windows)), key=lambda i: -windows[i].score):
        if len(chosen) == budget or windows[index].score < 1 / 255:
            break
        if distances[index] >= 4000:
            choose(index, windows[index].source)

    while len(chosen) < budget and np.any(distances >= 4000):
        choose(int(np.argmax(distances)), "keepalive")
    return [chosen[i] for i in sorted(chosen)]


def propose_windows(video_path: str | Path, *, budget: int, start_sec: float = 0.0,
                    end_sec: float | None = None, uniform: bool = False,
                    excluded: tuple[tuple[float, float], ...] = (), time_source: str = "pts",
                    pixel_attention: bool = False) -> list[StateWindow]:
    """Return up to budget chronological 2–4 s windows (shorter for short inputs).

    Seconds and indices are local to the decoded file. All three evidence frames
    fall inside the half-open interval. Any overlap with an excluded interval
    disqualifies the window. Uniform selection stratifies the same eligible windows.
    Pixel attention selects non-overlapping 4 s windows and samples center ±1 s.
    """
    if type(budget) is not int or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if time_source not in ("pts", "fps_formula_avi_cfr"):
        raise ValueError("unsupported frame-state time source")
    if not math.isfinite(start_sec) or start_sec < 0 or (
        end_sec is not None and (not math.isfinite(end_sec) or end_sec <= start_sec)
    ):
        raise ValueError("invalid video interval")
    if any(not math.isfinite(a) or not math.isfinite(b) or b <= a for a, b in excluded):
        raise ValueError("invalid excluded interval")
    if uniform and pixel_attention:
        raise ValueError("uniform selection and pixel attention are mutually exclusive")
    if budget == 0:
        return []
    times, observations, end = _scan(video_path, start_sec, math.inf if end_sec is None else end_sec, time_source)
    if pixel_attention:
        return _pixel_windows(times, observations, start_sec, end, min(budget, 30), excluded)
    windows = {}
    centers = {}
    for stamp, score, source, roi in observations:
        left, right = max(start_sec, stamp - 2), min(end, stamp + 2)
        if any(a < right and left < b for a, b in excluded):
            continue
        first, last = bisect_left(times, left), bisect_left(times, right) - 1
        mid = min(last, bisect_left(times, (left + right) / 2))
        indices = (first, mid, last)
        if indices in windows:
            continue
        if score < 1 / 255:
            source, roi = "keepalive", (0.0, 0.0, 1.0, 1.0)
        windows[indices] = StateWindow(left, right, indices, tuple(times[i] for i in indices),
                                       tuple(map(float, roi)), score, source)
        centers[indices] = stamp

    candidates = list(windows.values())
    sample_times = np.asarray([centers[w.frame_indices] for w in candidates])
    limit = min(budget, len(candidates))
    if not limit:
        return []
    if uniform:
        return [replace(candidates[(2 * slot + 1) * len(candidates) // (2 * limit)], source="uniform")
                for slot in range(limit)]
    distances = np.full(len(candidates), np.inf)
    chosen = {}

    def choose(i, source):
        chosen[i] = replace(candidates[i], source=source)
        np.minimum(distances, np.abs(sample_times - sample_times[i]), out=distances)
        distances[i] = -1

    # Keep uniform coverage under the same budget, about one window per 20 s.
    reserved = min(limit // 3, math.ceil((end - start_sec) / 20))
    for slot in range(reserved):
        target = start_sec + (slot + 0.5) * (end - start_sec) / reserved
        idx = int(np.argmin(np.where(distances >= 0, np.abs(sample_times - target), np.inf)))
        choose(idx, "keepalive")
    for i in sorted(range(len(candidates)), key=lambda j: -candidates[j].score):
        if len(chosen) == limit:
            break
        if candidates[i].source != "keepalive" and distances[i] >= 2:
            choose(i, candidates[i].source)
    # With little signal, spend the remaining slots covering the largest gaps.
    while len(chosen) < limit:
        i = int(np.argmax(distances)) if chosen else int(np.argmin(np.abs(sample_times - (start_sec + end) / 2)))
        choose(i, "keepalive")
    return sorted(chosen.values(), key=lambda w: (w.start_sec, w.end_sec))
