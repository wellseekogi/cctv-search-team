import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np

from embedding.contracts import SCHEMA, write_jsonl
from team_search.inputs import _b_samples, _selected_manifest
from modules.b_detection_tracking.src.b_video_io import CCTVFrameReader


class TeamInputsTests(unittest.TestCase):
    def test_ordered_reader_opens_once_per_video_and_matches_single_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / "frames.mkv"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("ffv1", rate=10)
                stream.width, stream.height, stream.pix_fmt = 16, 16, "bgr0"
                for i in range(4):
                    pixels = np.full((16, 16, 3), i * 40, dtype=np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            manifest, config = root / "manifest.jsonl", root / "vision.yaml"
            write_jsonl(manifest, [{"video_id": video_id, "relative_path": video.name,
                                   "available_start_ms": 1000, "available_end_ms": 1400,
                                   "source_offset_ms": 1000} for video_id in ("a", "b")])
            config.write_text("vision: {max_frame_error_ms: 100}", encoding="utf-8")
            reader = CCTVFrameReader(manifest, config)
            requests = [("a", 1050), ("a", 1050), ("a", 1100), ("a", 1350), ("b", 1000)]
            expected_indices = [0, 0, 1, 3, 0]
            with patch.object(av, "open", wraps=av.open) as open_video, reader.ordered() as read:
                frames = [read(*request) for request in requests]
                self.assertEqual(open_video.call_count, 2)
            for request, frame, index in zip(requests, frames, expected_indices):
                self.assertEqual(frame["frame_index"], index)
                self.assertEqual(frame["actual_timestamp_ms"], 1000 + 100 * index)
                np.testing.assert_array_equal(frame["rgb"], reader.read_frame(*request)["rgb"])

    def test_b_export_keeps_observation_provenance_and_requires_matching_frame(self):
        frame = {"schema_version": SCHEMA, "video_id": "camera01", "frame_index": 1,
                 "timestamp_ms": 200}
        track = {**frame, "source": "pred", "track_source_id": "litedetr01",
                 "local_track_id": "t000001", "track_uid": "litedetr01:camera01:t000001",
                 "class_name": "person", "confidence": 0.9, "bbox": [1, 2, 7, 8],
                 "quality": {"area_ratio": 0.1, "blur_var": 0.2, "border_touch": False}}
        with tempfile.TemporaryDirectory() as temp:
            frames = Path(temp) / "frames.jsonl"
            tracks = Path(temp) / "tracks.jsonl"
            write_jsonl(frames, [{**frame, "decode_status": "success"},
                                 {**frame, "frame_index": 2, "timestamp_ms": 400,
                                  "decode_status": "failed"}])
            write_jsonl(tracks, [track])
            scene, observation = _b_samples(frames, tracks, "t3-test-run")
            self.assertEqual((scene["representation"], scene["timestamp_ms"], scene["bbox"]),
                             ("scene", 200, None))
            self.assertEqual(observation["representation"], "object")
            self.assertEqual(observation["source_run_id"], "t3-test-run")
            self.assertEqual({key: observation[key] for key in track}, track)
            write_jsonl(tracks, [{**track, "frame_index": 2, "timestamp_ms": 400}])
            with self.assertRaisesRegex(ValueError, "no matching successfully decoded frame"):
                list(_b_samples(frames, tracks, "t3-test-run"))

    def test_legacy_manifest_only_selects_referenced_cameras(self):
        camera = {"video_id": "camera01", "relative_path": "camera01/vdo.avi"}
        other = {"video_id": "camera02", "relative_path": "camera02/vdo.avi"}
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "manifest.jsonl"
            samples = Path(temp) / "samples.jsonl"
            write_jsonl(manifest, [camera, other])
            write_jsonl(samples, [camera])
            self.assertEqual(_selected_manifest(manifest, samples), [{**camera, "schema_version": SCHEMA}])
            write_jsonl(manifest, [{**camera, "schema_version": "cctv-0"}])
            with self.assertRaisesRegex(ValueError, "Unsupported manifest schema"):
                _selected_manifest(manifest, samples)
