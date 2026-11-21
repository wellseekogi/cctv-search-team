"""Score F queries on full scene galleries and the submitted D API on the eval hybrid gallery."""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.app import create_app
from embedding.contracts import file_sha256, read_jsonl
from evaluation.benchmark_meva_search import load_boxes, localized_person, relevant_frame
from rag.engine import FrameRecommender
from retrieval.config import load_config
from retrieval.service import build_state


def first_rank(rows, predicate):
    return next((i for i, row in enumerate(rows, 1) if predicate(row)), None)


def reference_png(query, directory, media_path):
    if directory:
        png = (directory / Path(query["image_path"]).name).read_bytes()
    else:
        import cv2
        source = query["source"]
        cap = cv2.VideoCapture(str(media_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, source["frame_index"])
            ok, frame = cap.read()
        finally:
            cap.release()
        if not ok:
            raise ValueError("Cannot read reference frame: " + query["query_id"])
        x, y, right, bottom = source["crop_box"]
        ok, encoded = cv2.imencode(".png", frame[y:bottom, x:right])
        if not ok:
            raise ValueError("Cannot encode reference crop")
        png = encoded.tobytes()
    if hashlib.sha256(png).hexdigest() != query["image_sha256"]:
        raise ValueError("Reference image checksum differs: " + query["query_id"])
    return png


def score_pairs(path, scenes, encoder):
    frames = {}
    for engine in scenes.values():
        for item in engine.items:
            camera = item["video_id"].split(".")[-1]
            frames[f"{camera}_{item['timestamp_ms'] // 1000}"] = (engine, item)
    results = []
    for line in path.read_text(encoding="utf8").splitlines():
        if not line or line.startswith("#"):
            continue
        kind, positive, negative, ko, en = line.split("|")
        result = {"kind": kind, "positive": positive, "negative": negative, "ko": ko, "en": en}
        for lang, prompt in (("ko", ko), ("en", en)):
            vec = encoder.encode_text([prompt])
            scores = []
            for frame in (positive, negative):
                engine, item = frames[frame]
                scores.append(engine.ranked_frames(vec, [item], 1)[0]["ranking_score"])
            result[lang + "_margin"] = scores[0] - scores[1]
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "dev-scenes", "eval-scenes", "geometry", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--images", type=Path, help="F reference PNGs. If omitted, reproduce them from original media with OpenCV.")
    parser.add_argument("--pairs", type=Path, help="Optional labeled comparisons for the two-frame diagnostic")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_config(args.config)
    data = Path(__file__).resolve().parents[1] / "data/meva"
    ko = json.loads((data.parents[1] / "evaluation/meva_ko.json").read_text(encoding="utf8"))
    state = build_state(config)
    if state.health()["status"] != "ok":
        raise RuntimeError(state.health())
    gallery = state.galleries["meva_eval_v1"].engine.gallery
    aliases = {k: m["f_video_id"] for k, m in gallery.videos.items()}
    encoder = state.encoder.encoder
    scenes = {s: FrameRecommender(path, encoder) for s, path in (("dev", args.dev_scenes), ("eval", args.eval_scenes))}
    if any(engine.region_vectors is None for engine in scenes.values()):
        raise ValueError("Scene comparison requires four region vectors per frame")
    by_item = {r["item_id"]: r for r in gallery.items}
    # D can return a B observation box on a representative scene as well as on a crop.
    box_frames = [dict(r, video_id=aliases[r["video_id"]], bbox=True) for r in gallery.items]
    manifests = read_jsonl(data / "manifest.jsonl")
    media = {m["video_id"]: gallery.config.data_root / m["relative_path"] for m in manifests}
    hashes = {v: file_sha256(path) for v, path in media.items()}
    if hashes != {m["video_id"]: m["file_sha256"] for m in manifests}:
        raise ValueError("Source media checksum mismatch")
    for split, engine in scenes.items():
        if engine.index["sources"] != {m["video_id"]: hashes[m["video_id"]] for m in manifests if m["split"] == split}:
            raise ValueError("Scene index sources do not match the split")
    ms = [m for m in manifests if m["split"] == "eval"]
    boxes = load_boxes(args.geometry, ms, box_frames)
    labels = read_jsonl(data / "qrels.jsonl")
    queries = read_jsonl(data / "queries.jsonl")
    report = {"model": encoder.metadata["model_id"], "revision": encoder.metadata["model_revision"],
              "scene_counts": {s: len(e.items) for s, e in scenes.items()},
              "hybrid_items": len(by_item), "tracks": len(gallery.observations),
              "observations": sum(map(len, gallery.observations.values())), "source_sha256": hashes,
              "scene_index_sha256": {s: e.index_hash for s, e in scenes.items()}, "queries": []}
    excluded = [q["exclusion"] for q in queries if q["modality"] == "image"]
    for frame in [r for e in scenes.values() for r in e.items] + box_frames:
        if any(x["video_id"] == frame["video_id"] and x["start_ms"] <= frame["timestamp_ms"] < x["end_ms"] for x in excluded):
            raise ValueError("Reference image exclusion violated")

    with TestClient(create_app(config, state)) as client:
        for q in queries:
            qid, split = q["query_id"], q["split"]
            gt = [r for r in labels if r["query_id"] == qid and r["relevance"] > 0]
            if not gt:
                raise ValueError("Missing ground truth: " + qid)
            is_text = q["modality"] == "text"
            if is_text:
                prompts = [("ko", ko[qid]), ("en", q["text"])]
            else:
                png = reference_png(q, args.images, media[q["source"]["video_id"]])
                with Image.open(io.BytesIO(png)) as image:
                    rgb = np.asarray(image.convert("RGB"))
                prompts = [("image", None)]
            engine = scenes[split]
            for lang, prompt in prompts:
                vec = encoder.encode_text([prompt]) if is_text else encoder.encode_image([rgb])
                row = {"id": qid, "split": split, "kind": q["kind"], "language": lang}
                for mode, use_regions in (("full", False), ("scene", is_text)):
                    ranked = engine.ranked_frames(vec, engine.items, len(engine.items), text_search=use_regions)
                    row[mode] = first_rank(ranked, lambda h: relevant_frame(h, gt))
                if split == "eval":
                    fields = {"gallery_id": gallery.gallery_id, "representation": "combined", "top_k": 5}
                    if is_text:
                        response = client.post("/search", json=dict(fields, text=prompt))
                    else:
                        response = client.post("/search/image", data=fields, files={"image": ("q.png", png, "image/png")})
                    row["api_status"] = response.status_code
                    if response.status_code == 200:
                        payload = response.json()
                        hits = [dict(by_item[h["item_id"]], video_id=aliases[h["video_id"]], bbox=h["bbox"],
                                     timestamp_ms=h["representative_ms"]) for h in payload["results"]]
                        row["track_time"] = first_rank(hits, lambda h: relevant_frame(h, gt))
                        row["track_person"] = first_rank(hits, lambda h: localized_person(h, gt, boxes))
                        row["top5_items"] = [h["item_id"] for h in hits]
                    else:
                        row["api_error"] = response.json()
                    row["time_available"] = any(relevant_frame(h, gt) for h in box_frames)
                report["queries"].append(row)
    if args.pairs:
        report["pairs"] = score_pairs(args.pairs, scenes, encoder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(f'{len(report["queries"])} query/language results; {report["scene_counts"]}; {report["hybrid_items"]} hybrid items')


if __name__ == "__main__":
    main()
