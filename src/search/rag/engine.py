import heapq
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from embedding.contracts import check_vectors, file_sha256, read_jsonl
from embedding.pipeline import validate_index

from .errors import RagError
from .prompts import fallback_summary, validate_summary


def scene_context(vectors, rows_by_video, timestamps):
    context = {}
    for video, rows in rows_by_video.items():
        # The background correction was validated on dense, one-second scene galleries.
        stamps = np.unique(timestamps[rows])
        if len(stamps) < 30 or np.median(np.diff(stamps)) > 1000:
            continue
        ids = np.asarray(rows)
        batches = np.array_split(ids, (len(ids) + 255) // 256)
        mean = sum(vectors[b].sum(axis=0, dtype=np.float64) for b in batches) / len(ids)
        mean = mean.astype(np.float32)
        norms = np.concatenate([np.linalg.norm(vectors[b] - mean, axis=1) for b in batches])
        context[video] = (ids, mean, norms)
    return context


def text_ranking_scores(cosines, query_vector, context, object_videos=()):
    ranking = cosines.copy()
    for video, (ids, mean, norms) in context.items():
        if video not in object_videos:
            deviation = cosines[ids] - float(mean @ query_vector)
            ranking[ids] += 0.5 * np.divide(deviation, norms, out=np.zeros_like(deviation),
                                            where=norms > 1e-6)
    return ranking


class FrameRecommender:
    def __init__(self, index_dir, encoder, llm=None):
        self.root = Path(index_dir).resolve()
        self.encoder = encoder
        self.llm = llm
        self.index = validate_index(self.root, encoder.metadata)
        self.items = read_jsonl(self.root / "items.jsonl")
        self.vectors = np.load(self.root / "vectors.npy", mmap_mode="r", allow_pickle=False)
        self.index_hash = file_sha256(self.root / "index_manifest.json")
        self.video_ids = set(self.index["sources"]) | {r["video_id"] for r in self.items}
        scene_rows = defaultdict(list)
        for row in self.items:
            stamp = row.get("actual_timestamp_ms", row["timestamp_ms"])
            offset = row.get("source_offset_ms", 0)
            if (type(row["frame_index"]) is not int or row["frame_index"] < 0 or
                    type(stamp) is not int or type(offset) is not int or not 0 <= offset <= stamp):
                raise ValueError("Index frame/time metadata is invalid")
            if row["representation"] == "scene":
                scene_rows[row["video_id"]].append(row["vector_row"])
        stamps = np.asarray([r.get("actual_timestamp_ms", r["timestamp_ms"]) for r in self.items])
        self.scene_context = scene_context(self.vectors, scene_rows, stamps)
        self.region_vectors = None
        views_path = self.root / "scene_views.npz"
        if views_path.is_file():
            with np.load(views_path, allow_pickle=False) as views:
                if str(views["index_sha256"]) != self.index_hash:
                    raise ValueError("Region vectors belong to a different index")
                regions = views["vectors"]
            expected = (len(self.items), 4, self.encoder.dim)
            if regions.shape != expected or any(r["representation"] != "scene" for r in self.items):
                raise ValueError("Expected four region vectors per scene")
            check_vectors(regions.reshape(-1, self.encoder.dim), len(self.items) * 4, self.encoder.dim)
            self.region_vectors = regions

    def health(self):
        return {"status": "ready", "model_ready": True, "index_ready": True,
                "llm_ready": bool(self.llm is not None and getattr(self.llm, "ready", True)),
                "row_count": len(self.items),
                "embedding_model": self.encoder.metadata["model_id"],
                "embedding_dim": self.encoder.dim, "index_sha256": self.index_hash}

    def ranked_frames(self, query, candidates, top_k, *, text_search=True):
        recommendations = []
        check_vectors(query, 1, self.encoder.dim)
        scores = np.clip(self.vectors @ query[0], -1, 1)
        ranking = scores
        if text_search and self.region_vectors is not None:
            ranking = (scores + (self.region_vectors @ query[0]).max(axis=1)) * 0.5
        elif text_search and self.scene_context:
            mixed = {r["video_id"] for r in candidates if r["representation"] == "object"}
            ranking = text_ranking_scores(scores, query[0], self.scene_context, mixed)
        best = {}
        for row in candidates:
            stamp = row.get("actual_timestamp_ms", row["timestamp_ms"])
            score = float(scores[row["vector_row"]])
            order = (-float(ranking[row["vector_row"]]), row["video_id"], row["frame_index"], row["item_id"])
            key = (row["video_id"], row["frame_index"], stamp)
            if key not in best or order < best[key][0]:
                best[key] = (order, row, stamp, score)
        for order, row, stamp, score in heapq.nsmallest(top_k, best.values(), key=lambda v: v[0]):
            offset = row.get("source_offset_ms", 0)
            recommendations.append({"rank": len(recommendations) + 1, "item_id": row["item_id"],
                                    "video_id": row["video_id"], "frame_index": row["frame_index"],
                                    "timestamp_ms": stamp, "seconds": stamp / 1000,
                                    "source_offset_ms": offset, "playback_time_ms": stamp - offset,
                                    "score": score, "ranking_score": -order[0], "representation": row["representation"],
                                    "bbox": row["bbox"]})
        return recommendations

    def recommend(self, question, top_k=5, video_ids=None, representation="combined", use_llm=True):
        started = time.perf_counter()
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1000:
            raise RagError("INVALID_QUESTION", "question must contain 1 to 1000 characters")
        question = question.strip()
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise RagError("INVALID_TOP_K", "top_k must be an integer from 1 to 20")
        if representation not in ("scene", "object", "combined"):
            raise RagError("INVALID_REPRESENTATION", "Use scene, object, or combined")
        if type(use_llm) is not bool:
            raise RagError("INVALID_INPUT", "use_llm must be boolean")
        if video_ids is not None:
            if (not isinstance(video_ids, list) or
                    any(not isinstance(v, str) or not v for v in video_ids)):
                raise RagError("INVALID_VIDEO_IDS", "video_ids must be a string list or null")
            if set(video_ids) - self.video_ids:
                raise RagError("UNKNOWN_VIDEO", "A requested video is not registered in this index")
        allowed = None if video_ids is None else set(video_ids)
        candidates = [r for r in self.items if (allowed is None or r["video_id"] in allowed)
                      and (representation == "combined" or r["representation"] == representation)]
        recommendations = []
        if candidates:
            if hasattr(self.encoder, "text_token_lengths"):
                if self.encoder.text_token_lengths([question])[0] > self.encoder.text_max_length:
                    raise RagError("TEXT_TOO_LONG", f"Embedding input exceeds {self.encoder.text_max_length} tokens")
            try:
                query = self.encoder.encode_text([question])
            except (ValueError, RuntimeError) as exc:
                if "too long" in str(exc).lower() or "TEXT_TOO_LONG" in str(exc):
                    raise RagError("TEXT_TOO_LONG", "Question exceeds the embedding encoder token limit") from exc
                raise
            recommendations = self.ranked_frames(query, candidates, top_k)
        if not recommendations:
            summary, llm_status = "검색 가능한 프레임 후보가 없습니다.", "skipped_empty"
        else:
            summary = fallback_summary(len(recommendations))
            llm_status = "disabled" if not use_llm else "fallback"
            if use_llm and self.llm is not None:
                try:
                    summary = validate_summary(self.llm.summarize(question, len(recommendations)))
                    llm_status = "generated"
                except Exception:
                    # Frame recommendations remain usable when the optional narrator fails.
                    pass
        lines = [summary]
        for row in recommendations:
            lines.append(f"{row['rank']}. 영상 {row['video_id']}, 프레임 {row['frame_index']} "
                         f"(0부터 시작), {row['seconds']:.3f}초, 유사도 {row['score']:.4f}")
        return {"status": "success", "question": question, "summary": summary,
                "answer": "\n".join(lines), "recommendations": recommendations, "llm_status": llm_status,
                "retrieval": "scene_region_fusion" if self.region_vectors is not None else "text_frame_ranking_v2",
                "score_type": "cosine",
                "index_sha256": self.index_hash, "latency_ms": (time.perf_counter() - started) * 1000}
