"""Local test data preparation; not part of the product or handoff package."""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from embedding import build_index, create_encoder, load_config, validate_index
from embedding.contracts import SCHEMA, file_sha256, normalized, read_jsonl, write_json, write_jsonl
from rag.engine import FrameRecommender

ANNOTATIONS = "https://raw.githubusercontent.com/eembc/mlmark/main/datasets/COCO2017/annotations/instances_val2017.json"
CLASSES = {3: "car", 6: "bus", 8: "truck", 2: "bicycle", 4: "motorcycle", 10: "traffic light"}
QUERIES = ["버스가 있는 장면", "자전거가 있는 장면", "트럭이 있는 장면"]


def download(url, path):
    if not path.exists():
        with urlopen(url, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)


def select_photos(data):
    images = {row["id"]: row for row in data["images"] if row["license"] == 4}
    selected, used = [], set()
    for category, name in CLASSES.items():
        candidates = sorted((row for row in data["annotations"]
                             if row["category_id"] == category and not row["iscrowd"]
                             and row["image_id"] in images), key=lambda row: row["id"])
        count = 0
        for box in candidates:
            image = images[box["image_id"]]
            x, y, w, h = box["bbox"]
            min_side, min_area = (12, 0.001) if category == 10 else (24, 0.02)
            if image["id"] in used or min(w, h) < min_side or w * h < image["width"] * image["height"] * min_area:
                continue
            selected.append({"image": image, "annotation": box, "category": name})
            used.add(image["id"])
            count += 1
            if count == 5:
                break
        if count != 5:
            raise ValueError(f"Need five distinct usable CC BY photos for {name}; found {count}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/traffic30-real")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    photos = root / "photos"
    photos.mkdir(exist_ok=True)
    print("[1/6] Selecting 30 distinct real traffic photos (CC BY 2.0)...", flush=True)
    annotation_path = root / "coco_annotations.json"
    download(ANNOTATIONS, annotation_path)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    selected = select_photos(data)

    def get_photo(row):
        name = f"{row['image']['id']:012}.jpg"
        download(f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{name}", photos / name)
        with Image.open(photos / name) as image:
            image.load()
            if image.size != (row["image"]["width"], row["image"]["height"]):
                raise ValueError("Downloaded photo dimensions differ from COCO metadata")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(get_photo, selected))
    manifests, samples, sources = [], [], []
    for row in selected:
        image, box = row["image"], row["annotation"]
        video_id = f"coco-{image['id']:012}"
        relative = f"photos/{image['id']:012}.jpg"
        sha = file_sha256(root / relative)
        manifests.append({"schema_version": SCHEMA, "video_id": video_id, "dataset": "coco-val2017-still-photos",
                          "relative_path": relative, "width": image["width"], "height": image["height"],
                          "duration_ms": 1, "available_start_ms": 0, "available_end_ms": 1,
                          "source_offset_ms": 0, "split": "dev", "file_sha256": sha})
        x, y, w, h = box["bbox"]
        for representation in ("scene", "object"):
            samples.append({"schema_version": SCHEMA, "item_id": f"{video_id}-{representation}",
                            "video_id": video_id, "frame_index": 0, "timestamp_ms": 0,
                            "representation": representation,
                            "track_uid": f"coco-annotation-{box['id']}" if representation == "object" else None,
                            "bbox": [x, y, min(x + w, image["width"]), min(y + h, image["height"])] if representation == "object" else None,
                            "event_ids": [], "reasons": ["COCO_GT_SINGLE_IMAGE"], "policy": "demo",
                            "source_run_id": "coco-val2017-traffic30", "track_source_id": "coco-ground-truth"})
        sources.append({**row, "video_id": video_id, "local_path": relative, "sha256": sha,
                        "license": next(value for value in data["licenses"] if value["id"] == image["license"])})
    assert len(sources) == len({row["sha256"] for row in sources}) == 30
    write_jsonl(root / "manifest.jsonl", manifests)
    write_jsonl(root / "samples.jsonl", samples)
    write_json(root / "sources.json", {"annotation_url": ANNOTATIONS,
                                      "annotation_sha256": file_sha256(annotation_path), "photos": sources})
    registered = {row["video_id"]: row for row in manifests}

    def read_frame(video_id, timestamp_ms):
        if timestamp_ms != 0:
            raise ValueError("Still photos do not have video timestamps")
        row = registered[video_id]
        with Image.open(root / row["relative_path"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        return {"image_rgb": rgb, "actual_timestamp_ms": 0, "frame_index": 0,
                "width": row["width"], "height": row["height"], "source_offset_ms": 0,
                "time_source": "still_image_no_video_time"}

    print("[2/6] Loading the real SigLIP2 model...", flush=True)
    config = load_config("configs/embedding/embedding.siglip2.yaml")
    encoder = create_encoder(config)
    index = root / "index"
    print("[3/6] RGB photos -> full scenes / GT crops -> 60 image embeddings...", flush=True)
    if not (index / "index_manifest.json").exists():
        build_index(root / "samples.jsonl", root / "manifest.jsonl", root, index, root / "cache",
                    read_frame, encoder, "coco-still-rgb-v1", margin=config["crop_margin_ratio"],
                    batch_size=config["batch_size_gpu"] if encoder.device.startswith("cuda") else config["batch_size_cpu"])
    validate_index(index, encoder.metadata)
    items = read_jsonl(index / "items.jsonl")
    vectors = np.load(index / "vectors.npy", allow_pickle=False)
    assert vectors.shape == (60, 768)
    assert {row["video_id"] for row in items} == set(registered)

    print("[4/6] Exposing preprocessing, raw features and L2 normalization...", flush=True)
    with Image.open(index / items[0]["media_path"]) as image:
        batch = encoder.preprocess(images=[image.copy()], return_tensors="pt")
    pixels = batch["pixel_values"].to(encoder.device, dtype=encoder.dtype)
    with encoder.torch.inference_mode():
        raw = encoder.model.get_image_features(pixel_values=pixels).float().cpu().numpy()
    first = normalized(raw, encoder.dim)
    cosine = float(first[0] @ vectors[0])
    assert cosine >= 0.99999, f"Re-encoded vector differs: cosine={cosine}"
    queries = encoder.encode_text(QUERIES)
    scores = queries @ vectors.T
    engine = FrameRecommender(index, encoder)
    results = [engine.recommend(question, top_k=3, use_llm=False) for question in QUERIES]
    for q, result in enumerate(results):
        assert len({row["video_id"] for row in result["recommendations"]}) == 3
        assert np.isclose(result["recommendations"][0]["score"], scores[q].max(), atol=1e-4)
    norms = np.linalg.norm(vectors, axis=1)
    write_json(root / "embedding_report.json", {
        "counts": {"real_photos": 30, "scenes": 30, "object_crops": 30, "image_vectors": 60,
                   "query_vectors": len(QUERIES), "dimensions": 768, "dtype": str(vectors.dtype)},
        "encoder": encoder.metadata,
        "first_item_preprocessing": {"item_id": items[0]["item_id"], "pixel_tensor_shape": list(pixels.shape),
                                     "pixel_dtype": str(pixels.dtype), "raw_features": raw[0].tolist(),
                                     "raw_norm": float(np.linalg.norm(raw[0])), "normalized": first[0].tolist(),
                                     "stored_vector_cosine": cosine},
        "max_l2_norm_error": float(np.max(np.abs(norms - 1))),
        "image_vectors": [{"item_id": item["item_id"], "vector_row": item["vector_row"],
                           "representation": item["representation"], "media_path": item["media_path"],
                           "values": vectors[item["vector_row"]].tolist()} for item in items],
        "queries": [{"text": question, "token_length": encoder.text_token_lengths([question])[0],
                     "values": queries[q].tolist(), "scores": scores[q].tolist(), "result": results[q]}
                    for q, question in enumerate(QUERIES)],
        "limitations": ["Real still photos, not CCTV video. Zero timestamps are adapter placeholders.",
                        "GT boxes are supplied by COCO, not produced by a detector.",
                        "This is a demonstration, not a retrieval accuracy or tracking benchmark."]})
    with (root / "vectors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["item_id", "vector_row"] + [f"d{d}" for d in range(768)])
        writer.writerows([[item["item_id"], item["vector_row"], *vectors[item["vector_row"]]] for item in items])

    print("[5/6] Writing a photo contact sheet and VS Code preview...", flush=True)
    sheet = Image.new("RGB", (5 * 256, 6 * 186), "white")
    draw = ImageDraw.Draw(sheet)
    for n, row in enumerate(sources):
        left, top = n % 5 * 256, n // 5 * 186
        with Image.open(root / row["local_path"]) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (248, 156))
            sheet.paste(thumb, (left + (256 - thumb.width) // 2, top))
        draw.text((left + 4, top + 158), f"{n+1:02} {row['category']} {row['image']['id']:012}", fill="black")
    sheet.save(root / "contact_sheet.jpg", quality=90)
    lines = ["# 실제 교통 사진 30장 임베딩 확인", "",
             "실제 사진 **30장** → 전체 이미지 **30개** + GT 객체 crop **30개** → **60 × 768 float32** 벡터.", "",
             "**독립된 사진입니다. 0초는 어댑터용 값이며 실제 CCTV의 영상 시각이 아닙니다.** 객체 박스는 COCO 정답 주석입니다.", "",
             "모음의 영어 라벨은 선택에 사용한 COCO 주석입니다. 한 사진에 여러 종류의 대상이 함께 있을 수 있습니다.", "",
             "![실제 교통 사진 30장](contact_sheet.jpg)", "", "## 실제 임베딩 과정", "",
             f"첫 항목: `{items[0]['item_id']}`", "",
             f"RGB 이미지 → 모델 전처리 `{list(pixels.shape)}` / `{pixels.dtype}` → raw 768차원 → L2 정규화 → 저장.", "",
             f"정규화 전 길이: `{np.linalg.norm(raw[0]):.8f}`. 재계산·저장 벡터 코사인: `{cosine:.8f}`.", "",
             f"저장 벡터의 최대 L2 길이 오차: `{np.max(np.abs(norms-1)):.10f}`.", "",
             "첫 벡터의 앞 16개 값:", "", "```text", np.array2string(vectors[0, :16], precision=7), "```", "",
             "[전체 60개 벡터와 질문·전처리 값 JSON](embedding_report.json) · [768차원 전체 CSV](vectors.csv) · [출처·라이선스](sources.json)", "",
             "JSON의 `first_item_preprocessing`에서 정규화 전후, `image_vectors`에서 전체 벡터, `queries`에서 질문 벡터와 유사도를 볼 수 있습니다.", ""]
    by_id = {row["item_id"]: row for row in items}
    for result in results:
        lines += [f"## {result['question']}", ""]
        for candidate in result["recommendations"]:
            item = by_id[candidate["item_id"]]
            lines += [f"{candidate['rank']}. `{candidate['video_id']}`, `{candidate['representation']}`, 코사인 `{candidate['score']:.5f}`", "",
                      f"![추천 이미지](index/{item['full_frame_path']})", ""]
    lines += ["검색 점수는 정답 확률이 아닙니다. 추천 사진과 실제 대상을 직접 대조하세요. COCO 라벨만으로 한국어 검색 정확도를 보증하지 않습니다.", "",
              "출처: [COCO](https://cocodataset.org/#download). 선택한 사진은 COCO 메타데이터상 CC BY 2.0입니다. 원본 URL과 라이선스는 sources.json에 보존했습니다.", "",
              "현재 결과는 검색 흐름 확인용이며 실제 CCTV 검색 정확도·추적 성능은 측정하지 않았습니다.", ""]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("[6/6] PASS: 30 unique photos, 60 x 768 vectors, hashes/norms, re-encoding and 3 queries.")
    print("First stored vector (first 16 values):", vectors[0, :16])
    print("All 768 values per vector:", root / "embedding_report.json")
    print("VS Code Markdown preview:", root / "README.md")


if __name__ == "__main__":
    main()
