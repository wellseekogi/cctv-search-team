import gzip
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import av
import numpy as np
from PIL import Image

from embedding.contracts import SCHEMA, file_sha256, normalized, read_jsonl, write_jsonl
from embedding.track_input import (SequentialVideoReader, iter_embedding_batches,
                                   iter_jsonl, iter_track_batches, validate_track)
from tools.embed_tracks import run


def observation(frame_index=0, local_id="t000001", **changes):
    return {"schema_version": SCHEMA, "source": "pred", "track_source_id": "det01",
            "video_id": "v1", "frame_index": frame_index, "timestamp_ms": frame_index * 100,
            "local_track_id": local_id, "track_uid": f"det01:v1:{local_id}", "class_name": "car",
            "bbox": [10.2, 10.5, 30.8, 40.1], "confidence": 0.8,
            "quality": {"area_ratio": 0.05, "blur_var": 100, "border_touch": False}, **changes}


class Encoder:
    dim = 3
    device = "cpu"
    metadata = {"model_id": "test-only", "embedding_dim": 3}

    def encode_image(self, images):
        return normalized([np.asarray(image).mean(axis=(0, 1)) + 1 for image in images], self.dim)


class TrackInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []

    def reader(self, video_id, timestamp):
        self.calls.append((video_id, timestamp))
        return {"rgb": np.full((80, 100, 3), [200, 10, 30], dtype=np.uint8),
                "actual_timestamp_ms": timestamp, "frame_index": timestamp // 100,
                "width": 100, "height": 80, "source_offset_ms": 0, "time_source": "pts"}

    def test_original_bbox_quality_and_rgb_order_preserved(self):
        row = observation()
        items, images = next(iter_track_batches([row], self.reader))
        self.assertEqual(items[0]["bbox"], row["bbox"])
        self.assertEqual(items[0]["quality"], row["quality"])
        self.assertNotIn("score", items[0]["quality"])
        self.assertEqual(items[0]["crop_bbox"], [8, 7, 33, 44])
        self.assertEqual(images[0].mode, "RGB")
        self.assertEqual(images[0].getpixel((0, 0)), (200, 10, 30))
        self.assertEqual(images[0].size, (25, 37))
        self.assertNotIn("item_id", row)

    def test_optional_score_and_larger_source_id(self):
        row = observation(track_source_id="det01_b_colab_001_000",
                          track_uid="det01_b_colab_001_000:v1:t000001")
        row["quality"]["score"] = 0.71
        items, _ = next(iter_track_batches([row], self.reader))
        self.assertEqual(items[0]["quality"]["score"], 0.71)

    def test_lazy_bounded_batches_and_same_frame_across_batch_boundary(self):
        consumed = []

        def source():
            for index in range(10000):
                consumed.append(index)
                yield observation(local_id=f"t{index:06d}")

        batches = iter_track_batches(source(), self.reader, batch_size=3)
        self.assertEqual(len(next(batches)[0]), 3)
        self.assertEqual(consumed, [0, 1, 2])
        next(batches)
        self.assertEqual(len(consumed), 6)
        self.assertEqual(self.calls, [("v1", 0)])
        batches.close()

    def test_tail_empty_and_filter_before_decode(self):
        rows = [observation(i) for i in range(5)]
        self.assertEqual([len(items) for items, _ in iter_track_batches(rows, self.reader, batch_size=2)], [2, 2, 1])
        self.assertEqual(list(iter_track_batches([], self.reader)), [])
        self.calls.clear()
        self.assertEqual(list(iter_track_batches([observation(confidence=0.2)], self.reader, min_confidence=0.25)), [])
        self.assertEqual(self.calls, [])

    def test_reject_invalid_schema_numbers_identity_quality_and_bbox(self):
        for changes in ({"schema_version": "old"}, {"source": "gt"}, {"frame_index": True},
                        {"timestamp_ms": -1}, {"confidence": float("nan")}, {"confidence": 1.1},
                        {"track_uid": "wrong"}, {"quality": {}}, {"bbox": [1, 2, 1, 3]},
                        {"bbox": [-1, 0, 5, 5]}, {"bbox": [0, 0, float("inf"), 5]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_track(observation(**changes))
        with self.assertRaisesRegex(ValueError, "outside original frame"):
            next(iter_track_batches([observation(bbox=[0, 0, 101, 80])], self.reader))

    def test_exact_frame_timestamp_dimensions_and_color_contract(self):
        for changes in ({"frame_index": 1}, {"actual_timestamp_ms": 1}, {"width": 99},
                        {"source_offset_ms": -1}, {"time_source": ""},
                        {"rgb": np.zeros((80, 100, 3), np.float32)}):
            def reader(*args):
                return {**self.reader(*args), **changes}
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                next(iter_track_batches([observation()], reader))

    def test_c_image_rgb_contract_also_accepted(self):
        def reader(*args):
            frame = self.reader(*args)
            frame["image_rgb"] = Image.fromarray(frame.pop("rgb"))
            return frame
        self.assertEqual(len(next(iter_track_batches([observation()], reader))[1]), 1)

    def test_embeddings_are_aligned_and_invalid_vectors_stop(self):
        items, vectors = next(iter_embedding_batches([observation(), observation(2)], self.reader, Encoder()))
        self.assertEqual([row["frame_index"] for row in items], [0, 2])
        self.assertEqual(vectors.shape, (2, 3))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=0.0001)
        encoder = Encoder()
        encoder.encode_image = lambda images: np.zeros((len(images), 3), np.float32)
        with self.assertRaisesRegex(ValueError, "Normalization"):
            next(iter_embedding_batches([observation()], self.reader, encoder))

    def test_jsonl_gzip_bom_blank_and_bad_line(self):
        path = self.root / "tracks.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8-sig") as stream:
            stream.write("\n" + json.dumps(observation()) + "\n[]\n")
        rows = iter_jsonl(path)
        self.assertEqual(next(rows)["frame_index"], 0)
        with self.assertRaisesRegex(ValueError, r":3:.*JSON object"):
            next(rows)
        bad = self.root / "bad.jsonl"
        bad.write_text('{"broken":\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, ":1:"):
            next(iter_jsonl(bad))

    def test_invalid_batch_settings(self):
        for options in ({"batch_size": 0}, {"batch_size": True}, {"margin": -1},
                        {"margin": float("nan")}, {"min_confidence": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                next(iter_track_batches([], self.reader, **options))

    def make_video(self):
        path = self.root / "video.mkv"
        with av.open(str(path), "w") as container:
            stream = container.add_stream("ffv1", rate=10)
            stream.width, stream.height, stream.pix_fmt = 100, 80, "bgr0"
            for index in range(5):
                array = np.full((80, 100, 3), [index * 20, 30, 200], np.uint8)
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        self.video = {"video_id": "v1", "width": 100, "height": 80, "source_offset_ms": 0,
                      "available_start_ms": 0, "available_end_ms": 500, "split": "dev",
                      "file_sha256": file_sha256(path)}
        self.frames = [{"schema_version": SCHEMA, "video_id": "v1", "frame_index": i,
                        "timestamp_ms": i * 100, "width": 100, "height": 80, "source_offset_ms": 0,
                        "time_source": "pts", "decode_status": "success"} for i in range(5)]
        return path

    def test_sequential_decode_skips_unneeded_frames_and_reuses_pixels(self):
        path = self.make_video()
        with SequentialVideoReader(path, self.video, self.frames) as reader:
            first = reader.read_frame("v1", 0)
            self.assertIs(reader.read_frame("v1", 0), first)
            frame = reader.read_frame("v1", 300)
            self.assertEqual(frame["frame_index"], 3)
            self.assertEqual(frame["rgb"][0, 0].tolist(), [60, 30, 200])
            self.assertEqual(next(reader.decoded)[0], 4)
            with self.assertRaisesRegex(ValueError, "backwards"):
                reader.read_frame("v1", 100)
        self.assertIsNone(reader.container)

    def test_sequential_offset_and_mismatched_pts(self):
        path = self.make_video()
        self.video.update(source_offset_ms=1000, available_start_ms=1000, available_end_ms=1500)
        for row in self.frames:
            row["timestamp_ms"] += 1000
            row["source_offset_ms"] = 1000
        with SequentialVideoReader(path, self.video, self.frames) as reader:
            self.assertEqual(reader.read_frame("v1", 1200)["frame_index"], 2)
        self.frames[2]["timestamp_ms"] += 1
        with SequentialVideoReader(path, self.video, self.frames) as reader:
            with self.assertRaisesRegex(ValueError, "FRAME_MISMATCH"):
                reader.read_frame("v1", 1201)

    def test_missing_failed_and_wrong_manifest_frames_rejected(self):
        path = self.make_video()
        cases = [[], [{**self.frames[0], "decode_status": "failed"}],
                 [{**self.frames[0], "width": 101}], [{**self.frames[0], "video_id": "other"}],
                 [{**self.frames[0], "frame_index": 50}]]
        for frames in cases:
            with self.subTest(frames=frames), SequentialVideoReader(path, self.video, frames) as reader:
                with self.assertRaises(ValueError):
                    reader.read_frame("v1", 0)

    def cli_args(self):
        path = self.make_video()
        write_jsonl(self.root / "manifest.jsonl", [self.video])
        write_jsonl(self.root / "frames.jsonl", self.frames)
        write_jsonl(self.root / "tracks.jsonl", [observation(i) for i in range(5)])
        return Namespace(config="configs/embedding/embedding.siglip2.yaml", manifest=self.root / "manifest.jsonl",
                         video_id="v1", video=path, tracks=self.root / "tracks.jsonl",
                         frames=self.root / "frames.jsonl", output=self.root / "out",
                         batch_size=2, shard_size=2, min_confidence=0)

    def test_cli_shards_row_mapping_and_overwrite_protection(self):
        args = self.cli_args()
        status = run(args, encoder=Encoder())
        self.assertEqual((status["row_count"], status["parts"]), (5, 3))
        self.assertEqual(json.loads((args.output / "manifest.json").read_text())["status"], "success")
        for index, count in enumerate([2, 2, 1]):
            part = args.output / f"part-{index:06d}"
            self.assertEqual(np.load(part / "vectors.npy").shape, (count, 3))
            self.assertEqual([row["vector_row"] for row in read_jsonl(part / "items.jsonl")], list(range(count)))
        with self.assertRaises(FileExistsError):
            run(args, encoder=Encoder())

    def test_shards_accumulate_multiple_model_batches(self):
        args = self.cli_args()
        args.shard_size = 3
        status = run(args, encoder=Encoder())
        self.assertEqual((status["row_count"], status["parts"]), (5, 2))
        self.assertEqual(np.load(args.output / "part-000000/vectors.npy").shape, (4, 3))
        self.assertEqual(np.load(args.output / "part-000001/vectors.npy").shape, (1, 3))

    def test_cli_source_hash_failure_and_partial_run_not_published(self):
        args = self.cli_args()
        self.video["file_sha256"] = "bad"
        write_jsonl(args.manifest, [self.video])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            run(args, encoder=Encoder())
        self.assertFalse(args.output.exists())
        self.video["file_sha256"] = file_sha256(args.video)
        write_jsonl(args.manifest, [self.video])
        write_jsonl(args.tracks, [observation(0), observation(1), observation(2, bbox=[0, 0, 0, 1])])
        with self.assertRaises(ValueError):
            run(args, encoder=Encoder())
        self.assertTrue((args.output / "part-000000/vectors.npy").is_file())
        self.assertTrue((args.output / "failure.json").is_file())
        self.assertFalse((args.output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
