
import json
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

    def read_frame(self, video_id, requested_timestamp_ms):
        if video_id not in self.manifest_by_id:
            raise FrameReadError(
                "VIDEO_NOT_FOUND", video_id
            )

        m = self.manifest_by_id[video_id]
        requested_timestamp_ms = int(
            requested_timestamp_ms
        )

        start = int(m["available_start_ms"])
        end = int(m["available_end_ms"])

        if not start <= requested_timestamp_ms < end:
            raise FrameReadError(
                "OUT_OF_RANGE",
                str(requested_timestamp_ms)
            )

        offset = int(m.get("source_offset_ms") or 0)
        max_error = int(
            self.config["vision"]["max_frame_error_ms"]
        )

        path = self.video_path_for(m)

        best = None
        first_pts = None

        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]

                for idx, frame in enumerate(
                    container.decode(stream)
                ):
                    if (
                        frame.pts is not None
                        and frame.time_base is not None
                    ):
                        pts = float(
                            frame.pts * frame.time_base
                        )

                        if first_pts is None:
                            first_pts = pts

                        timestamp_ms = offset + round(
                            (pts - first_pts) * 1000
                        )
                        time_source = "pts"
                    else:
                        rate = stream.average_rate
                        if rate is None or float(rate) <= 0:
                            raise RuntimeError(
                                "FPS unavailable for PTS fallback"
                            )

                        timestamp_ms = offset + round(
                            idx * 1000 / float(rate)
                        )
                        time_source = "estimated"

                    distance = abs(
                        timestamp_ms - requested_timestamp_ms
                    )

                    if best is None or (
                        distance, timestamp_ms
                    ) < (
                        best["distance"],
                        best["timestamp_ms"]
                    ):
                        best = {
                            "frame_index": idx,
                            "timestamp_ms": timestamp_ms,
                            "time_source": time_source,
                            "distance": distance,
                            "rgb": frame.to_ndarray(
                                format="rgb24"
                            )
                        }

                    if timestamp_ms > requested_timestamp_ms:
                        break

        except FrameReadError:
            raise
        except Exception as e:
            raise FrameReadError(
                "DECODE_FAILED", repr(e)
            ) from e

        if best is None or best["distance"] > max_error:
            raise FrameReadError(
                "FRAME_NOT_FOUND",
                str(requested_timestamp_ms)
            )

        rgb = best["rgb"]

        return {
            "rgb": rgb,
            "actual_timestamp_ms": best["timestamp_ms"],
            "frame_index": best["frame_index"],
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "time_source": best["time_source"],
            "source_offset_ms": offset
        }


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
