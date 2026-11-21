"""Measure timestamp retrieval on held-out MEVA recordings and optional hybrid index person localization."""

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import torch

from embedding import create_encoder
from embedding.contracts import file_sha256, read_jsonl
from evaluation.common import iou
from integrations.f_metadata import image_exclusions
from rag.engine import FrameRecommender, scene_context, text_ranking_scores
from team_search.__main__ import ROOT, get_config


def scene_gallery(manifests, queries, media, encoder):
    items, vectors, batch = [], [], []
    for m in manifests:
        path = media / m["relative_path"]
        if file_sha256(path) != m["file_sha256"] or m["fps"] != 30:
            raise ValueError(f"Unexpected MEVA source: {path}")
        excluded = image_exclusions(queries, dict(m, f_video_id=m["video_id"]))
        with av.open(str(path)) as video:
            stream = video.streams.video[0]
            for frame_no, frame in enumerate(video.decode(stream)):
                # F's AVI qrels use decoded frame indices on the original CFR timeline.
                ms = round(1000 * frame_no / m["fps"])
                if frame_no % 30 or any(e["start_ms"] <= ms < e["end_ms"] for e in excluded):
                    continue
                items.append({"video_id": m["video_id"], "timestamp_ms": ms, "frame_index": frame_no})
                batch.append(frame.to_ndarray(format="rgb24"))
                if len(batch) == 4:
                    vectors.extend(encoder.encode_image(batch))
                    batch.clear()
            if frame_no + 1 != m["frame_count"]:
                raise ValueError(f"Incomplete decode: {path}")
        print("Decoded", m["camera_id"], frame_no + 1, flush=True)
    if batch:
        vectors.extend(encoder.encode_image(batch))
    return items, np.asarray(vectors)


def relevant_frame(hit, truths):
    vid = hit["video_id"].removeprefix("meva_")
    return [
        t for t in truths if vid == t["video_id"] and t["start_ms"] <= hit["timestamp_ms"] < t["end_ms"]
    ]


def rank_stats(flags):
    first = next((i for i, match in enumerate(flags, 1) if match), None)
    return {"rank": first, "relevant": sum(flags), "candidates": len(flags)}


def localized_person(hit, truths, boxes):
    if not hit["bbox"]:
        return False
    annotated = boxes.get((hit["video_id"].removeprefix("meva_"), hit["frame_index"]), {})
    return any(
        t["gt_track_uid"] in annotated and iou(hit["bbox"], annotated[t["gt_track_uid"]]) >= 0.5
        for t in relevant_frame(hit, truths)
    )


def load_boxes(folder, manifests, items):
    boxes = defaultdict(dict)
    wanted = {(i["video_id"].removeprefix("meva_"), i["frame_index"]) for i in items if i["bbox"]}
    for m in manifests:
        path = folder / (m["camera_id"] + ".geom.yml")
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if "'geom':" not in line:
                    continue
                geom = ast.literal_eval(line.strip().removeprefix("- "))["geom"]
                key = m["video_id"], geom["ts0"]
                if key in wanted:
                    boxes[key][f"gtm1:{key[0]}:m{geom['id1']:04d}"] = list(map(float, geom["g0"].split()))
    if not all(k in boxes for k in wanted):
        raise ValueError("Missing geometry for an indexed person frame")
    return boxes


def benchmark(args):
    data = ROOT / "data/meva"
    qs = read_jsonl(data / "queries.jsonl")
    qrels = read_jsonl(data / "qrels.jsonl")
    tests = [q for q in qs if q["split"] == "eval" and q["modality"] == "text"]
    manifests = [m for m in read_jsonl(data / "manifest.jsonl") if m["split"] == "eval"]
    ko = json.loads((ROOT / "evaluation/meva_ko.json").read_text(encoding="utf-8"))
    encoder = create_encoder(get_config(args))
    frames, vectors = scene_gallery(manifests, qs, args.media, encoder)
    by_video = defaultdict(list)
    for i, frame in enumerate(frames):
        by_video[frame["video_id"]].append(i)
    stamps = np.asarray([f["timestamp_ms"] for f in frames])
    context = {} if args.baseline else scene_context(vectors, by_video, stamps)
    hybrid = FrameRecommender(args.hybrid_index, encoder) if args.hybrid_index else None
    boxes = load_boxes(args.geometry, manifests, hybrid.items) if args.geometry else {}
    rows = []
    for q in tests:
        truths = [r for r in qrels if r["query_id"] == q["query_id"] and r["relevance"] > 0]
        if not truths:
            raise ValueError("Missing qrels for " + q["query_id"])
        for lang, prompt in (("en", q["text"]), ("ko", ko[q["query_id"]])):
            query = encoder.encode_text([prompt])
            scores = text_ranking_scores(vectors @ query[0], query[0], context)
            ordered = np.argsort(-scores, kind="stable")
            flags = [bool(relevant_frame(frames[i], truths)) for i in ordered]
            row = {
                "query": q["query_id"],
                "language": lang,
                "full_scene": rank_stats(flags),
                "first_scene": frames[ordered[0]],
            }
            if hybrid:
                ranked = hybrid.ranked_frames(query, hybrid.items, len(hybrid.items), text_search=not args.baseline)
                row["hybrid_frame"] = rank_stats([bool(relevant_frame(h, truths)) for h in ranked])
                if args.geometry:
                    row["hybrid_person"] = rank_stats([localized_person(h, truths, boxes) for h in ranked])
            rows.append(row)
    return {
        "model": encoder.metadata["model_id"],
        "revision": encoder.metadata["model_revision"],
        "gallery_frames": len(frames),
        "source_hashes": {m["camera_id"]: m["file_sha256"] for m in manifests},
        "queries": rows,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", action="store_true", help="Use the archived unadjusted cosine ranking")
    p.add_argument("--media", required=True, type=Path)
    p.add_argument("--hybrid-index", type=Path)
    p.add_argument("--geometry", type=Path, help="Pinned <camera>.geom.yml files for returned-box IoU")
    p.add_argument("--embedding-config", type=Path, default=ROOT / "configs/embedding.siglip2.yaml")
    p.add_argument("--model-root", type=Path)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.geometry and not args.hybrid_index:
        p.error("--geometry requires --hybrid-index")
    torch.set_num_threads(4)
    report = benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "Recorded", len(report["queries"]), "query/language pairs over", report["gallery_frames"], "frames"
    )


if __name__ == "__main__":
    main()
