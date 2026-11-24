
import json
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

import av
import yaml


class FrameReadError(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code = code


class CCTVFrameReader:
    def __init__(
        self,
        manifest_path,
        config_path,
        path_overrides=None
    ):
        self.manifest_path = Path(manifest_path)
        self.config_path = Path(config_path)

        with self.config_path.open(
            "r", encoding="utf-8"
        ) as f:
            self.config = yaml.safe_load(f)

        with self.manifest_path.open(
            "r", encoding="utf-8"
        ) as f:
            manifest = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        self.manifest_by_id = {
            m["video_id"]: m for m in manifest
        }

        self.path_overrides = {
            k: Path(v)
            for k, v in (path_overrides or {}).items()
        }

    def video_path_for(self, m):
        video_id = m["video_id"]

        if video_id in self.path_overrides:
            path = self.path_overrides[video_id]
            if not path.is_file():
                raise FrameReadError(
                    "DECODE_FAILED",
                    f"Video path not found: {path}"
                )
            return path

        relative = Path(m["relative_path"])

        project_root = self.manifest_path.parent.parent
        data_root = self.manifest_path.parent

        candidates = [
            relative,
            data_root / relative,
            project_root / relative,
            project_root / "data" / relative,
        ]

        for path in candidates:
            if path.is_file():
                return path

        raise FrameReadError(
            "DECODE_FAILED",
            f"No video path found for {video_id}"
        )

    def _decode(self, manifest, offset):
        first_pts = None
        with av.open(str(self.video_path_for(manifest))) as container:
            stream = container.streams.video[0]
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is not None and frame.time_base is not None:
                    pts = float(frame.pts * frame.time_base)
                    if first_pts is None:
                        first_pts = pts
                    yield offset + round((pts - first_pts) * 1000), index, "pts", frame
                else:
                    rate = stream.average_rate
                    if rate is None or float(rate) <= 0:
                        raise RuntimeError("FPS unavailable for PTS fallback")
                    yield offset + round(index * 1000 / float(rate)), index, "estimated", frame

    @contextmanager
    def ordered(self):
        """Reuse the decoder for successive requests in video and timestamp order."""
        active_video = last_requested = None
        decoded = previous = upcoming = None

        with ExitStack() as resources:
            def read(video_id, requested_timestamp_ms):
                nonlocal active_video, last_requested, decoded, previous, upcoming

                if video_id not in self.manifest_by_id:
                    raise FrameReadError("VIDEO_NOT_FOUND", video_id)
                manifest = self.manifest_by_id[video_id]
                requested = int(requested_timestamp_ms)
                if not int(manifest["available_start_ms"]) <= requested < int(manifest["available_end_ms"]):
                    raise FrameReadError("OUT_OF_RANGE", str(requested))
                offset = int(manifest.get("source_offset_ms") or 0)
                max_error = int(self.config["vision"]["max_frame_error_ms"])

                try:
                    if active_video != video_id or requested < last_requested:
                        resources.close()
                        decoded = resources.enter_context(closing(self._decode(manifest, offset)))
                        active_video = video_id
                        previous, upcoming = None, next(decoded, None)
                    last_requested = requested
                    while upcoming is not None and upcoming[0] <= requested:
                        # Keep the first frame when multiple frames have the same timestamp.
                        if previous is None or upcoming[0] > previous[0]:
                            previous = upcoming
                        upcoming = next(decoded, None)
                except FrameReadError:
                    active_video = None
                    raise
                except Exception as exc:
                    active_video = None
                    raise FrameReadError("DECODE_FAILED", repr(exc)) from exc

                candidates = (frame for frame in (previous, upcoming) if frame is not None)
                best = min(candidates, key=lambda frame: (abs(frame[0] - requested), frame[0]), default=None)
                if best is None or abs(best[0] - requested) > max_error:
                    raise FrameReadError("FRAME_NOT_FOUND", str(requested))
                timestamp, index, time_source, frame = best
                try:
                    rgb = frame.to_ndarray(format="rgb24")
                except Exception as exc:
                    raise FrameReadError("DECODE_FAILED", repr(exc)) from exc
                return {
                    "rgb": rgb,
                    "actual_timestamp_ms": timestamp,
                    "frame_index": index,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "time_source": time_source,
                    "source_offset_ms": offset,
                }

            yield read

    def read_frame(self, video_id, requested_timestamp_ms):
        with self.ordered() as read:
            return read(video_id, requested_timestamp_ms)


_default_reader = None


def configure_reader(
    manifest_path,
    config_path,
    path_overrides=None
):
    global _default_reader
    _default_reader = CCTVFrameReader(
        manifest_path,
        config_path,
        path_overrides
    )
    return _default_reader


def read_frame(video_id, requested_timestamp_ms):
    if _default_reader is None:
        raise RuntimeError(
            "Call configure_reader() first"
        )

    return _default_reader.read_frame(
        video_id,
        requested_timestamp_ms
    )
