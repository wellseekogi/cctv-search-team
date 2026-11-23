import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from sampling.frame_state import propose_windows


class FrameStateTests(unittest.TestCase):
    def make_video(self, folder, *, patch=False, pts=None, filename="scene.mkv"):
        path = Path(folder) / filename
        with av.open(str(path), "w") as dst:
            stream = dst.add_stream("ffv1", rate=4)
            stream.width, stream.height = 128, 96
            stream.pix_fmt = "gray"
            for i, stamp in enumerate(pts if pts is not None else range(80)):
                # Uniform exposure jump at 4 s. An optional stationary patch appears at 12 s.
                pixels = np.full((96, 128), 20 if i < 16 else 85, dtype=np.uint8)
                if patch and i >= 48:
                    pixels[32:48, 96:112] = 220
                frame = av.VideoFrame.from_ndarray(pixels, format="gray")
                frame.time_base = Fraction(1, 4)
                frame.pts = stamp
                for packet in stream.encode(frame):
                    dst.mux(packet)
            for packet in stream.encode():
                dst.mux(packet)
        return path

    def test_untracked_change_and_persistent_state_survive_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_video(tmp, patch=True)
            windows = propose_windows(path, budget=5)
            self.assertEqual(windows, propose_windows(path, budget=5))
            self.assertEqual(len(windows), 5)
            strongest = max(windows, key=lambda w: w.score)
            self.assertEqual(strongest.source, "change")
            self.assertTrue(strongest.start_sec < 12 < strongest.end_sec)
            x1, y1, x2, y2 = strongest.roi_xyxy
            self.assertTrue(x1 <= 104/128 <= x2 and y1 <= 40/96 <= y2)
            self.assertTrue(any(w.source == "persistent" for w in windows))
            self.assertTrue(any(w.source == "keepalive" for w in windows))

    def test_exposure_only_uses_periodic_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_video(tmp)
            windows = propose_windows(path, budget=3)
            self.assertEqual(len(windows), 3)
            self.assertLessEqual(windows[0].start_sec, 2)
            self.assertGreaterEqual(windows[-1].end_sec, 18)
            self.assertTrue(all(w.score == 0 and w.source == "keepalive" for w in windows))
            self.assertEqual(propose_windows(path, budget=0), [])

    def test_clipped_variable_pts_and_short_eof(self):
        with tempfile.TemporaryDirectory() as tmp:
            # An intentional one-second PTS gap invalidates index/rate arithmetic.
            path = self.make_video(tmp, pts=[0, 1, 2, 3, 8, 9, 10, 11, 12])
            windows = propose_windows(path, budget=8, start_sec=2.1, end_sec=99)
            self.assertEqual(len(windows), 1)
            window = windows[0]
            self.assertEqual(window.frame_indices, (5, 7, 8))
            self.assertEqual(window.frame_times_sec, (2.25, 2.75, 3.0))
            self.assertAlmostEqual(window.start_sec, 2.1)
            self.assertAlmostEqual(window.end_sec, 3.25)
            with self.assertRaisesRegex(ValueError, "no decoded frames"):
                propose_windows(path, budget=1, start_sec=20)

    def test_uniform_control_shares_exclusion_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_video(tmp, patch=True)
            common = dict(budget=5, excluded=((0, 4.5), (17, 25)))
            by_signal = propose_windows(path, **common)
            uniform = propose_windows(path, uniform=True, **common)
            self.assertTrue(all(w.source == "uniform" for w in uniform))
            for windows in (by_signal, uniform):
                self.assertEqual(len(windows), 5)
                self.assertTrue(all(4.5 <= w.start_sec < w.end_sec <= 17 for w in windows))
            self.assertEqual(propose_windows(path, budget=10, excluded=((0, 20),)), [])

    def test_pixel_attention_uses_non_overlapping_windows_and_centered_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_video(tmp, patch=True)
            windows = propose_windows(path, budget=2, excluded=((0, 4.5),), pixel_attention=True)
            self.assertEqual(len(windows), 2)
            self.assertLessEqual(windows[0].end_sec, windows[1].start_sec)
            self.assertGreaterEqual(windows[0].start_sec, 4.5)
            self.assertTrue(all(w.end_sec - w.start_sec == 4 for w in windows))
            strongest = max(windows, key=lambda w: w.score)
            self.assertEqual(strongest.frame_times_sec, (11., 12., 13.))
            x1, y1, x2, y2 = strongest.roi_xyxy
            self.assertAlmostEqual(x2 - x1, 1 / 4)
            self.assertAlmostEqual(y2 - y1, 1 / 3)
            self.assertAlmostEqual((x1 + x2) / 2, 104 / 128)
            self.assertAlmostEqual((y1 + y2) / 2, 40 / 96)

    def test_pixel_ranking_uses_only_the_full_frame_top_sixty_for_fusion(self):
        from team_search.frame_state import rank_windows

        rows = [dict(video_id="camera", start_ms=i * 4000) for i in range(61)]
        vectors = np.zeros((61, 3, 2, 1), dtype=np.float32)
        vectors[:, :, 0, 0] = np.linspace(1, 0, 61)[:, None]
        vectors[59:, :, 1, 0] = 1
        query = np.ones(1, dtype=np.float32)
        ranked = rank_windows(rows, vectors, query, pixel_attention=True)
        self.assertEqual(ranked[0]["start_ms"], rows[59]["start_ms"])
        self.assertEqual(ranked[-1]["start_ms"], rows[60]["start_ms"])

    def test_avi_uses_pts_unless_manifest_declares_fixed_cadence(self):
        from team_search.frame_state import interval_rows

        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_video(tmp, pts=[0, 1, 2, 3, 8, 9, 10, 11, 12], filename="scene.avi")
            movie = dict(video_id="camera", source_offset_ms=0, available_start_ms=0, available_end_ms=4000)
            # This real AVI has a one-second gap in the timestamps of its nine frames.
            self.assertEqual(propose_windows(path, budget=10)[-1].frame_times_sec[-1], 3.0)
            for mode, expected_end in (("pts", 3000), ("fps_formula_avi_cfr", 2000)):
                rows = interval_rows(path, dict(movie, time_source=mode), [], 10, False)
                self.assertEqual(rows[-1]["frame_indices"][-1], 8)
                self.assertEqual(rows[-1]["frame_times_ms"][-1], expected_end)
