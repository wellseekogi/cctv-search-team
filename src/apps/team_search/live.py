"""Single-process loopback event ingestion, search and grounded chat."""
import argparse
import hashlib
import io
import json
import mimetypes
import os
import re
import socket
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from embedding import create_encoder
from embedding.contracts import within
from rag.assets import ensure_assets
from rag.runtime import LocalLLM
from .__main__ import ROOT, get_config
from .live_store import LiveStore
from .full import add_full_options, full_catalog, playback_file
from .demo import DEMO_KEY, DEMO_NAME

UI_ROOT = Path(__file__).resolve().parent.parent / "ui"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1200)


class Search(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=4)
    gallery_id: str | None = None
    video_ids: list[str] | None = None
    representation: Literal["combined", "scene", "object"] = "combined"
    top_k: int = Field(default=3, ge=1, le=50)


def create_app(args):
    generation = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        config = get_config(args)
        encoder = create_encoder(config)
        app.state.full = full_catalog(args, config, encoder) if getattr(args, "full", False) else {}
        asset_root = ROOT / "models/local-rag"
        sibling = ROOT.parent / "cctv-search/models/local-rag"
        if not asset_root.exists() and sibling.is_dir():
            asset_root = sibling
        assets = ensure_assets(ROOT / "configs/rag/rag-model.json", asset_root)
        with LocalLLM(assets["server_path"], assets["model_path"], port=args.llm_port,
                      log_dir=args.store, context=4096, output_tokens=256, request_timeout=150) as llm:
            app.state.llm = llm
            store = app.state.store = LiveStore(args.store, config, encoder)
            try:
                yield
            finally:
                store.close()

    app = FastAPI(title="CCTV 실시간 검색", lifespan=lifespan)

    @app.middleware("http")
    async def local_only(request, call_next):
        host = request.headers.get("host", "")
        if host.split(":")[0] not in {"localhost", "127.0.0.1"}:
            return JSONResponse({"detail": "로컬 주소를 사용하세요."}, status_code=403)
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != host:
            return JSONResponse({"detail": "같은 주소에서 요청하세요."}, status_code=403)
        return await call_next(request)

    @app.get("/")
    def home():
        return RedirectResponse("/ui/")

    @app.get("/api/status")
    def status():
        store = app.state.store
        full = app.state.full
        return {"llm_ready": app.state.llm.ready, "model": "Qwen3.5-4B",
                "embedding": store.encoder.metadata["model_id"], "jobs": store.status(),
                "default_gallery": "gallery_all" if full else "gallery_dev",
                "videos": len(full), "vectors": sum(len(engine.items) for engine, _ in (full or store.catalog).values())}

    def enqueue(kind, key, name):
        try:
            app.state.store.submit(kind, key, name)
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc
        return {"id": key}

    @app.post("/api/events/demo", status_code=202)
    def demo():
        return enqueue("demo", DEMO_KEY, DEMO_NAME)

    @app.post("/api/events/handoff/{batch}", status_code=202)
    def handoff(batch: str):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch):
            raise HTTPException(422, "배치 이름이 올바르지 않습니다.")
        root = within(app.state.store.root / "inbox", batch)
        if not root.is_dir():
            raise HTTPException(404, "인계 폴더가 없습니다.")
        return enqueue("handoff", "b-" + batch, batch)

    @app.post("/api/events/clip", status_code=202)
    async def clip(request: Request, name: str = "video.mp4"):
        folder = app.state.store.root / "uploads"
        digest, size = hashlib.sha256(), 0
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
            path = Path(tmp.name)
        try:
            with path.open("wb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > 128 * 1024 * 1024:
                        raise HTTPException(413, "구간당 최대 128 MiB입니다.")
                    digest.update(chunk)
                    output.write(chunk)
            if not size:
                raise HTTPException(422, "빈 영상입니다.")
            key = digest.hexdigest()
            target = folder / key
            if not target.exists():
                os.replace(path, target)
            return enqueue("clip", key, Path(name).name[:160] or "video.mp4")
        finally:
            path.unlink(missing_ok=True)

    def retrieve(query, image=None):
        started = time.perf_counter()
        store = app.state.store
        gallery = query.gallery_id or ("gallery_all" if app.state.full else "gallery_dev")
        if gallery == "gallery_all" and app.state.full:
            catalog = app.state.full
        elif gallery == "gallery_dev":
            catalog = store.catalog
        else:
            raise HTTPException(404, "등록되지 않은 데이터셋입니다.")
        if not catalog:
            raise HTTPException(409, "먼저 영상을 등록하고 검색 가능 상태를 확인하세요.")
        allowed = set(query.video_ids or [])
        if allowed - {v for engine, _ in catalog.values() for v in engine.video_ids}:
            raise HTTPException(404, "등록되지 않은 영상 ID입니다.")
        text = query.text.strip()
        if not text:
            raise HTTPException(422, "질문을 입력하세요.")
        with store.encoder_lock:
            try:
                vector = (store.encoder.encode_text([text]) if image is None
                          else store.encoder.encode_image([image]))
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        results, total = [], 0
        for key, (engine, media) in catalog.items():
            candidates = [r for r in engine.items if (not allowed or r["video_id"] in allowed)
                          and query.representation in ("combined", r["representation"])]
            total += len(candidates)
            lookup = {r["item_id"]: r for r in candidates}
            for hit in engine.ranked_frames(vector, candidates, query.top_k, text_search=image is None):
                r = lookup[hit["item_id"]]
                base = f"/media/{key}/{r['vector_row']}"
                stamp = hit["timestamp_ms"]
                results.append(dict(**hit, track_uid=r["track_uid"], width=r["width"], height=r["height"],
                                    source_sha256=engine.index["sources"][r["video_id"]],
                                    display_name=media[r["video_id"]][2],
                                    representative_ms=stamp, start_ms=stamp,
                                    end_ms=min(stamp + 2000, media[r["video_id"]][1]),
                                    thumbnail_url=base + "/frame", video_url=base + "/video",
                                    event_ids=r["event_ids"], reasons=r["reasons"]))
        results.sort(key=lambda r: (-r["ranking_score"], r["video_id"], r["frame_index"], r["thumbnail_url"]))
        unique = {}
        for row in results:
            frame = (row["source_sha256"], row["frame_index"], row["playback_time_ms"])
            unique.setdefault(frame, row)
            if len(unique) == query.top_k:
                break
        results = list(unique.values())
        for rank, row in enumerate(results, 1):
            row["rank"] = rank
        version = hashlib.sha256("".join(sorted(catalog)).encode()).hexdigest()[:10]
        return dict(request_id=uuid4().hex[:10], index_version=version, total_candidates=total,
                    score_type="cosine", results=results, latency_ms=round((time.perf_counter()-started)*1000))

    @app.post("/search")
    def search(query: Search):
        started = time.perf_counter()
        reply = retrieve(query)
        if not reply["results"]:
            return dict(reply, answer="해당 조건에 맞는 검색 후보가 없습니다.", llm_status="skipped_empty")
        evidence = [{"순위": r["rank"], "프레임": r["frame_index"],
                     "재생_초": r["playback_time_ms"]/1000}
                    for r in reply["results"][:5]]
        with generation:
            try:
                answer = app.state.llm.chat(query.text, [t.model_dump() for t in query.history], evidence)
            except (OSError, KeyError, ValueError, RuntimeError) as exc:
                raise HTTPException(503, "Qwen 답변 생성에 실패했습니다. 다시 질문해 주세요.") from exc
        reply.update(answer=answer, llm_status="generated", latency_ms=round((time.perf_counter()-started)*1000))
        return reply

    @app.post("/search/image")
    def search_image(image: UploadFile = File(), gallery_id: str | None = Form(None),
                     top_k: int = Form(3, ge=1, le=50), representation: str = Form("combined"),
                     video_ids: str = Form("[]")):
        raw = image.file.read(5000001)
        if len(raw) > 5000000:
            raise HTTPException(413, "이미지는 최대 5 MB입니다.")
        try:
            query = Search(text="이미지", top_k=top_k, gallery_id=gallery_id,
                           video_ids=json.loads(video_ids), representation=representation)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format not in ("JPEG", "PNG") or min(im.size) < 32 or im.width * im.height > 16000000:
                    raise ValueError("이미지 규격을 확인하세요.")
                return retrieve(query, im.convert("RGB"))
        except (ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise HTTPException(422, "이미지 또는 검색 조건이 올바르지 않습니다.") from exc

    @app.get("/media/{key}/{index}/{kind}")
    def media(key: str, index: int, kind: Literal["frame", "video"]):
        shard = app.state.full.get(key) or app.state.store.catalog.get(key)
        if not shard or not 0 <= index < len(shard[0].items):
            raise HTTPException(404)
        engine, videos = shard
        row = engine.items[index]
        if kind == "frame":
            return FileResponse(within(engine.root, row["full_frame_path"]), media_type="image/png")
        path = videos[row["video_id"]][0]
        if key in app.state.full:
            path = playback_file(path, engine.root.parent)
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0] or "video/mp4")

    @app.get("/ui/config.js")
    def ui_config():
        script = (UI_ROOT / "config.js").read_text(encoding="utf-8")
        if app.state.full:
            script += '\nwindow.UI_CONFIG.default_gallery = "gallery_all";'
        return Response(script, media_type="application/javascript", headers={"Cache-Control": "no-cache"})

    app.mount("/ui", StaticFiles(directory=UI_ROOT, html=True), name="ui")
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--llm-port", type=int, default=8781)
    parser.add_argument("--store", type=Path, help="기존 이벤트 저장소 전용 모드")
    add_full_options(parser)
    parser.add_argument("--embedding-config", type=Path, default=ROOT / "configs/embedding/embedding.siglip2.yaml")
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"))
    args = parser.parse_args()
    args.full = args.store is None
    args.store = args.store or ROOT / "runs/live"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
        server = uvicorn.Server(uvicorn.Config(create_app(args), timeout_keep_alive=5))
        server.run(sockets=[sock])
