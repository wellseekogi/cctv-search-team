"""Ask questions and let the local LLM recommend three retrieved image names."""

import argparse
import json
from pathlib import Path

import numpy as np

from embedding import create_encoder, load_config, validate_index
from embedding.contracts import check_vectors, read_jsonl
from rag.assets import ensure_assets

from .image_answer import ImageAnswerLLM


def search(encoder, items, vectors, question, top_k):
    query = encoder.encode_text([question])
    check_vectors(query, 1, vectors.shape[1])
    scores = np.clip(vectors @ query[0], -1, 1)
    order = sorted(range(len(items)), key=lambda n: (-float(scores[n]), items[n]["item_id"]))
    return query[0], [(items[n], vectors[n], float(scores[n])) for n in order[:top_k]]


def interact(encoder, items, vectors, llm):
    print(f"\n로컬 RAG 테스트 | 사진 {len(items)}장 | 답변 생성: Qwen3.5-4B")
    print("질문을 입력하고 Enter를 누르세요. 종료: /exit 또는 종료")
    while True:
        try:
            question = input("\n질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return 0
        if question.lower() in ("/exit", "exit", "quit", "종료"):
            print("종료합니다.")
            return 0
        if not question:
            continue
        try:
            _, matches = search(encoder, items, vectors, question, 3)
        except ValueError as exc:
            print(f"질문 오류: {exc}")
            continue
        try:
            answer = llm.answer(question, [item["image_name"] for item, _, _ in matches])
        except (ValueError, RuntimeError, OSError) as exc:
            print(f"LLM 답변 생성 실패: {exc}")
            continue
        print(f"\n답변 > {answer}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="runs/traffic30-real/index")
    parser.add_argument("--llm-port", type=int, default=18766)
    args = parser.parse_args()
    root = Path(args.index).resolve()
    if not (root / "index_manifest.json").is_file():
        parser.error("Prepare the local data first: python -m devtools.traffic_demo")
    print("SigLIP2와 Qwen3.5-4B를 로딩합니다. 준비 후 질문을 직접 입력하세요.", flush=True)
    encoder = create_encoder(load_config("configs/embedding/embedding.siglip2.yaml"))
    validate_index(root, encoder.metadata)
    items = [row for row in read_jsonl(root / "items.jsonl") if row["representation"] == "scene"]
    if len(items) < 3:
        parser.error("이미지 추천을 위해 원본 사진이 최소 3장 필요합니다")
    sources = root.parent / "sources.json"
    names = {}
    if sources.is_file():
        names = {row["video_id"]: Path(row["local_path"]).name
                 for row in json.loads(sources.read_text(encoding="utf-8"))["photos"]}
    items = [{**row, "image_name": names.get(row["video_id"], Path(row["media_path"]).name)} for row in items]
    stored = np.load(root / "vectors.npy", allow_pickle=False)
    vectors = stored[[row["vector_row"] for row in items]]
    assets = ensure_assets("configs/rag/rag-model.json", "models/local-rag")
    with ImageAnswerLLM(assets["server_path"], assets["model_path"], port=args.llm_port,
                        log_dir="runs/traffic30-real/terminal-llm", output_tokens=256,
                        request_timeout=60) as llm:
        return interact(encoder, items, vectors, llm)


if __name__ == "__main__":
    raise SystemExit(main())
