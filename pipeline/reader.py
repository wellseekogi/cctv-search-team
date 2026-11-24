"""B의 frames.jsonl과 같은 시각 규칙으로 읽되, 영상을 한 번만 순차 디코딩하는 리더."""
import av


class FrameReadError(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}")
        self.code = code


class _Video:
    def __init__(self, path, offset, time_source=None):
        self.path, self.offset, self.container = path, offset, None
        self.time_source = time_source
        self.open()

    def open(self):
        self.close()
        self.container = av.open(str(self.path))
        stream = self.container.streams.video[0]
        self.rate = float(stream.average_rate or 0)
        self.frames = self.container.decode(stream)
        self.index, self.first_pts = 0, None
        self.cur = self.nxt = None
        self.done = False

    def close(self):
        if self.container is not None:
            self.container.close()
            self.container = None

    def pull(self):
        frame = next(self.frames, None)
        if frame is None:
            self.done = True
            return None
        if self.time_source and self.time_source.startswith("fps_formula"):
            if self.rate <= 0:
                raise FrameReadError("DECODE_FAILED", "FPS unavailable for fps formula")
            stamp = self.offset + round(self.index * 1000 / self.rate)
            source = self.time_source
        elif frame.pts is not None and frame.time_base is not None:
            pts = float(frame.pts * frame.time_base)
            if self.first_pts is None:
                self.first_pts = pts
            stamp, source = self.offset + round((pts - self.first_pts) * 1000), "pts"
        else:
            if self.rate <= 0:
                raise FrameReadError("DECODE_FAILED", "FPS unavailable for PTS fallback")
            stamp, source = self.offset + round(self.index * 1000 / self.rate), "estimated"
        item = (stamp, self.index, source, frame)
        self.index += 1
        return item

    def read(self, target, tolerance):
        if self.cur is not None and target < self.cur[0]:
            self.open()  # 뒤로 가는 요청만 처음부터 다시 읽는다
        while not self.done and (self.nxt is None or self.nxt[0] <= target):
            if self.nxt is not None:
                self.cur = self.nxt
            self.nxt = self.pull()
        candidates = [c for c in (self.cur, self.nxt) if c is not None]
        if not candidates:
            raise FrameReadError("FRAME_NOT_FOUND", str(target))
        stamp, index, source, frame = min(candidates, key=lambda c: (abs(c[0] - target), c[0]))
        if abs(stamp - target) > tolerance:
            raise FrameReadError("FRAME_NOT_FOUND", f"{target} (closest {stamp})")
        rgb = frame.to_ndarray(format="rgb24")
        return dict(image_rgb=rgb, actual_timestamp_ms=stamp, frame_index=index,
                    width=int(rgb.shape[1]), height=int(rgb.shape[0]),
                    time_source=source, source_offset_ms=self.offset)


class SequentialReader:
    """videos: {video_id: (path, source_offset_ms, time_source)}. 요청 시각은 영상별로 오름차순이어야 빠르다."""

    def __init__(self, videos, tolerance_ms=100):
        self.specs, self.tolerance, self.opened = dict(videos), tolerance_ms, {}

    def read_frame(self, video_id, timestamp_ms):
        if video_id not in self.specs:
            raise FrameReadError("VIDEO_NOT_FOUND", video_id)
        if video_id not in self.opened:
            self.opened[video_id] = _Video(*self.specs[video_id])
        return self.opened[video_id].read(int(timestamp_ms), self.tolerance)

    def close(self):
        for video in self.opened.values():
            video.close()
        self.opened = {}