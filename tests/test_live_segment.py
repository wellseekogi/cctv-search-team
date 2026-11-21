import tempfile
import time
import unittest
from pathlib import Path

import av
from PIL import Image

from embedding.contracts import file_sha256
from team_search.live_store import LiveStore
from tests.test_embedding import TestEncoder


class LiveSegmentTests(unittest.TestCase):
    def test_clip_without_tracks_survives_restart_and_duplicate_delivery(self):
        config = {"crop_margin_ratio": 0.1, "batch_size_cpu": 2}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temp:
            root = Path(temp)
            video = root / "upload.mp4"
            with av.open(str(video), "w") as output:
                stream = output.add_stream("libx264", rate=5)
                stream.width = stream.height = 64
                for color in ("black", "white"):
                    for packet in stream.encode(av.VideoFrame.from_image(Image.new("RGB", (64, 64), color))):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            key = file_sha256(video)
            store = LiveStore(root, config, TestEncoder())
            try:
                video.rename(root / "uploads" / key)
                for _ in range(2):
                    store.submit("clip", key, "test.mp4")
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    jobs = store.status()
                    if jobs[0]["state"] in ("ready", "failed"):
                        break
                    time.sleep(0.05)
                self.assertEqual(len(jobs), 1)
                self.assertEqual(jobs[0]["state"], "ready", jobs[0]["error"])
                self.assertGreater(jobs[0]["vectors"], 0)
            finally:
                store.close()
            restored = LiveStore(root, config, TestEncoder())
            try:
                engine, media = restored.catalog[key]
                self.assertEqual(len(engine.items), jobs[0]["vectors"])
                self.assertEqual(file_sha256(media[key][0]), key)
                del engine
            finally:
                restored.close()
