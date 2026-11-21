"""Add four overlapping image-region vectors to each row of a scene-only search index."""
import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from embedding import create_encoder
from embedding.contracts import check_vectors, file_sha256, read_jsonl, within
from embedding.pipeline import validate_index
from team_search.__main__ import ROOT, get_config

def image_regions(image):
    width, height = image.size
    w, h = (3 * width + 4) // 5, (3 * height + 4) // 5
    return [image.crop((x, y, x + w, y + h)) for y in (0, height - h) for x in (0, width - w)]

def save_scene_views(folder, index_hash, vectors):
    with tempfile.NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as output:
        path = Path(output.name)
    try:
        np.savez_compressed(path, index_sha256=index_hash, vectors=vectors)
        if file_sha256(folder / "index_manifest.json") != index_hash:
            raise ValueError("Index changed while encoding regions")
        os.replace(path, folder / "scene_views.npz")
    finally:
        path.unlink(missing_ok=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embedding-config", type=Path, default=ROOT / "configs/embedding.siglip2.yaml")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    encoder = create_encoder(get_config(args))
    index_hash = file_sha256(args.index / "index_manifest.json")
    validate_index(args.index, encoder.metadata)
    rows = read_jsonl(args.index / "items.jsonl")
    if not rows or any(row["representation"] != "scene" for row in rows):
        parser.error("Supply a nonempty scene-only index")
    vectors = []
    for row in rows:
        with Image.open(within(args.index, row["full_frame_path"])) as image:
            features = encoder.encode_image(image_regions(image.convert("RGB")))
        check_vectors(features, 4, encoder.dim)
        vectors.append(features)
    save_scene_views(args.index, index_hash, np.stack(vectors))
    print(f"Indexed four regions for {len(rows)} scenes")

if __name__ == "__main__":
    main()
