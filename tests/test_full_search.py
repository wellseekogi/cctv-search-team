import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
from PIL import Image

from embedding.contracts import SCHEMA, file_sha256, read_jsonl, write_jsonl
from team_search.full import full_catalog
from tests.test_embedding import TestEncoder as StubEncoder


class FullSearchTests(unittest.TestCase):
    def test_full_gallery_embeds_only_a_selection_across_splits(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            root = Path(tmp)
            args = SimpleNamespace(
                metadata_root=root / "data",
                data_root=root,
                b_inputs=root / "b",
                gallery_cache=root / "output",
            )
            for dataset, split, color in (("meva", "dev", "white"), ("virat", "eval", "black")):
                video_id = dataset + "-fixture"
                source = root / (dataset + ".mp4")
                with av.open(str(source), "w") as dst:
                    stream = dst.add_stream("libx264", rate=5)
                    stream.width = stream.height = 64
                    for _ in range(30):
                        frame = av.VideoFrame.from_image(Image.new("RGB", (64, 64), color))
                        for packet in stream.encode(frame):
                            dst.mux(packet)
                    for packet in stream.encode():
                        dst.mux(packet)
                video = dict(
                    schema_version=SCHEMA,
                    video_id=video_id,
                    dataset=dataset,
                    split=split,
                    width=64,
                    height=64,
                    frame_count=30,
                    fps=5,
                    duration_ms=6000,
                    source_offset_ms=0,
                    available_start_ms=0,
                    available_end_ms=6000,
                    relative_path=source.name,
                    file_sha256=file_sha256(source),
                )
                data = args.metadata_root / dataset
                data.mkdir(parents=True)
                write_jsonl(data / "manifest.jsonl", [video])
                write_jsonl(data / "queries.jsonl", [])
                b = args.b_inputs / dataset
                b.mkdir(parents=True)
                write_jsonl(b / "manifest.jsonl", [video])
                frames = [
                    dict(
                        schema_version=SCHEMA,
                        video_id=video_id,
                        frame_index=i,
                        timestamp_ms=i * 200,
                        requested_timestamp_ms=i * 200,
                        source_offset_ms=0,
                        time_source="pts",
                        decode_status="success",
                        detector_status="ran",
                        width=64,
                        height=64,
                    )
                    for i in range(30)
                ]
                write_jsonl(b / "frames.jsonl", frames)
                write_jsonl(b / "tracks.jsonl", [])
                write_jsonl(
                    b / "video_status.jsonl",
                    [dict(video_id=video_id, video_status="completed", available_end_ms=6000)],
                )
            config = dict(crop_margin_ratio=0.1, batch_size_cpu=2)
            encoder = StubEncoder()
            catalog = full_catalog(args, config, encoder)
            self.assertEqual(len(catalog), 2)
            for engine, _ in catalog.values():
                selected = read_jsonl(engine.root.parent / "samples.jsonl")
                self.assertEqual([s["item_id"] for s in selected], [s["item_id"] for s in engine.items])
                self.assertEqual([s["timestamp_ms"] for s in engine.items], [0, 5000])
                self.assertTrue(all(s["policy"] == "hybrid" for s in engine.items))
            del engine, catalog
            encoder.calls.clear()
            self.assertEqual(len(full_catalog(args, config, encoder)), 2)
            self.assertFalse(encoder.calls)
            write_jsonl(b / "frames.jsonl", frames[:10])
            with (
                patch("team_search.full.ensure_observations", return_value=b) as prepare,
                self.assertRaisesRegex(ValueError, "전체 구간"),
            ):
                full_catalog(args, config, encoder)
            prepare.assert_called_once()
