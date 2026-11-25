"""Prepare the annotated MEVA excerpt and a single playback test result for the D UI."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import av
import numpy as np

from embedding.contracts import file_sha256, write_jsonl
from retrieval.encoder import FixtureEncoder
from team_search.demo import ROOT, make_demo


def prepare(directory):
    make_demo(directory)
    tracks = [json.loads(line) for line in (directory / "tracks.jsonl").read_text().splitlines()]
    track = tracks[len(tracks) // 2]
    with av.open(str(directory / "G421-m0222.mp4")) as video:
        frame = next(frame for i, frame in enumerate(video.decode(video=0)) if i == track["frame_index"])
        image = frame.to_image()
    image.save(directory / "frame.jpg")
    crop = image.crop(tuple(round(x) for x in track["bbox"]))
    crop.save(directory / "person.jpg")
    item = {key: track[key] for key in ("video_id", "frame_index", "timestamp_ms", "track_uid", "bbox")}
    item.update(item_id="g421-person", representation="object", vector_row=0,
                full_frame_path="frame.jpg", media_path="person.jpg")
    write_jsonl(directory / "items.jsonl", [item])
    encoder = FixtureEncoder()
    np.save(directory / "vectors.npy", encoder.encode_image([np.asarray(crop)]))
    metadata = dict(model_id=encoder.model_id, model_revision=encoder.model_revision,
                    preprocess_hash=encoder.preprocess_hash, embedding_dim=encoder.embedding_dim,
                    dtype="float32", normalized=True, track_source="gt",
                    files={name: dict(sha256=file_sha256(directory / name), rows=1)
                           for name in ("items.jsonl", "vectors.npy")})
    (directory / "index_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")


if __name__ == "__main__":
    target = ROOT / "runs/d-avi-playback"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with TemporaryDirectory(dir=target.parent) as tmp:
            staging = Path(tmp) / "playback"
            prepare(staging)
            staging.rename(target)
    print(target / "G421-m0222.mp4")
