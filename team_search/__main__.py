import argparse
import sys
import tempfile
from pathlib import Path

from embedding import create_encoder, load_config
from embedding.contracts import read_jsonl
from rag.engine import FrameRecommender
from rag.errors import RagError


ROOT = Path(__file__).resolve().parents[1]


def get_config(args):
    config = load_config(args.embedding_config)
    if args.model_root is not None:
        config["model_root"] = args.model_root.resolve()
    else:
        root = ROOT / config["model_root"]
        previous = ROOT.parent / "cctv-search" / config["model_root"]
        config["model_root"] = previous if not root.exists() and previous.is_dir() else root
    if args.device is not None:
        config["device"] = args.device
    return config


def build(args, config, **inputs):
    from .inputs import build_index_from_team

    status = build_index_from_team(
        **inputs, cache=args.cache, embedding_config=config, vision_config=args.vision_config,
    )
    if status["status"] != "success":
        errors = read_jsonl(Path(inputs["output"]) / "errors.jsonl")
        codes = ", ".join(sorted({error["code"] for error in errors}))
        raise ValueError(f"인덱스 생성 실패 ({codes}). 출력 경로 {Path(inputs['output']).resolve()}")


def ask(args, engines, encoder):
    previews = {(row["video_id"], row["item_id"]): engine.root / row["full_frame_path"]
                for engine in engines for row in engine.items}

    def show(question):
        if len(engines) == 1:
            rows = engines[0].recommend(question, top_k=args.top_k, representation=args.representation,
                                        use_llm=False)["recommendations"]
        else:
            if not 1 <= len(question.strip()) <= 1000:
                raise RagError("INVALID_QUESTION", "질문은 1~1000자로 입력하세요.")
            vector = encoder.encode_text([question])
            rows = []
            for engine in engines:
                candidates = [r for r in engine.items if args.representation in ("combined", r["representation"])]
                rows.extend(engine.ranked_frames(vector, candidates, args.top_k))
            rows.sort(key=lambda r: (-r["ranking_score"], r["video_id"], r["frame_index"], r["item_id"]))
            rows = rows[:args.top_k]
        if not rows:
            print("검색 가능한 프레임 후보가 없습니다.")
        for rank, row in enumerate(rows, 1):
            time = f"재생 {row['playback_time_ms'] / 1000:.3f}초"
            if row["source_offset_ms"]:
                time += f" (원본 {row['seconds']:.3f}초)"
            score = f"유사도 {row['score']:.4f}"
            if row["ranking_score"] != row["score"]:
                score += f" | 순위 점수 {row['ranking_score']:.4f}"
            print(f"{rank}. {row['video_id']} | 프레임 {row['frame_index']} | {time} | {score}")
            print(f"   {previews[row['video_id'], row['item_id']]}")

    if args.question is not None:
        show(args.question)
        return

    print(f"검색 준비 완료 ({len(previews)}개 항목). '종료' 또는 /exit로 끝냅니다.")
    while True:
        try:
            question = input("질문 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question in ("종료", "/exit"):
            return
        if question:
            try:
                show(question)
            except (RagError, ValueError) as exc:
                print(f"질문 오류 {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="CCTV 장면 인덱스 생성과 자연어 검색")
    model = argparse.ArgumentParser(add_help=False)
    model.add_argument("--embedding-config", type=Path, default=ROOT / "configs/embedding.siglip2.yaml")
    model.add_argument("--model-root", type=Path)
    model.add_argument("--device", choices=("auto", "cpu", "cuda"))
    build_options = argparse.ArgumentParser(add_help=False)
    build_options.add_argument("--vision-config", type=Path,
                               default=ROOT / "modules/b_detection_tracking/configs/vision.yaml")
    build_options.add_argument("--cache", type=Path, default=ROOT / ".cache/embedding")
    ask_options = argparse.ArgumentParser(add_help=False)
    ask_options.add_argument("--question", help="생략하면 대화형 질의")
    ask_options.add_argument("--top-k", type=int, choices=range(1, 21), default=3, metavar="1–20")
    ask_options.add_argument("--representation", choices=("scene", "object", "combined"), default="combined")

    from .full import add_full_options, full_catalog

    commands = parser.add_subparsers(dest="command", required=True)
    full = commands.add_parser("full", parents=[model, ask_options], help="전체 영상의 A hybrid 선별 결과 검색")
    add_full_options(full)
    prepare = commands.add_parser("build", parents=[model, build_options], help="A 또는 B 산출물에서 인덱스 생성")
    for option in ("manifest", "data-root", "output"):
        prepare.add_argument(f"--{option}", required=True, type=Path)
    for option in ("samples", "frames", "tracks"):
        prepare.add_argument(f"--{option}", type=Path)
    query = commands.add_parser("ask", parents=[model, ask_options], help="기존 인덱스 질의")
    query.add_argument("--index", required=True, type=Path)
    demo = commands.add_parser("demo", parents=[model, build_options, ask_options], help="MEVA 카페 사람 영상으로 질의")
    demo.add_argument("--output", type=Path, default=ROOT / "runs/team-demo-meva-g421")
    arguments = sys.argv[1:]
    if not arguments or (arguments[0].startswith("-") and arguments[0] not in ("--help", "-h")):
        arguments = ["full", *arguments]
    args = parser.parse_args(arguments)

    try:
        config = get_config(args)
        if args.command == "build":
            build(args, config, manifest=args.manifest, data_root=args.data_root, output=args.output,
                  samples=args.samples, frames=args.frames, tracks=args.tracks)
            print(f"인덱스 준비 완료 {args.output.resolve()}")
            return 0

        print("검색 모델을 준비하고 있습니다…", flush=True)
        encoder = create_encoder(config)
        if args.command == "full":
            catalog = full_catalog(args, config, encoder)
            ask(args, [engine for engine, _ in catalog.values()], encoder)
            return 0
        if args.command == "ask":
            ask(args, [FrameRecommender(args.index, encoder)], encoder)
            return 0

        from .demo import DEMO_KEY, DEMO_NAME
        from .live_input import process_segment

        if not args.output.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".team-demo-", dir=args.output.parent) as temp:
                workspace = Path(temp) / "demo"
                workspace.mkdir()
                process_segment("demo", DEMO_KEY, DEMO_NAME, None, workspace, config, encoder,
                                cache=args.cache, vision_config=args.vision_config)
                workspace.rename(args.output)
        print("MEVA 카페 예시 질의 '회색 후드티를 입은 사람이 테이블을 떠나는 장면을 찾아줘'.")
        ask(args, [FrameRecommender(args.output / "index", encoder)], encoder)
        return 0
    except (KeyError, OSError, ValueError, RuntimeError, RagError) as exc:
        print(f"실행 오류 {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
