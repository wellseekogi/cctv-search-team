"""평가 실행기 (F 5~7, G 5 `pipeline evaluate` 가 호출).

사용 예:
  python evaluation/evaluate.py --dataset virat --split dev --run-dir runs/r001 --mode gt
  python evaluation/evaluate.py --dataset virat --split eval --run-dir runs/r002 --mode pred

run-dir 에서 읽는 파일 (없으면 해당 지표를 null로 기록):
  predictions.jsonl  D: 질의별 전체 트랙 순위
      {"query_id": "...", "representation": "combined", "results": [{"rank": 1, "track_uid": "...", "score": 0.03}, ...]}
  samples.jsonl      A: 선택한 scene·object
  items.jsonl        C: 인코딩 성공 항목 (item_id)
  tracks.jsonl       B: pred 관측 (--mode pred 에서 필요)
  frames.jsonl       B: 실제 프레임 시각 (pred-GT 판정 가능 시각, 선택)
출력: <run-dir>/metrics.json, <run-dir>/track_links.jsonl (pred), <run-dir>/eval_per_query.jsonl,
      <run-dir>/eval_targets.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, load_eval_config, read_jsonl, write_json, write_jsonl  # noqa: E402
from matching import link_tracks  # noqa: E402
from metrics import evaluate_rankings  # noqa: E402
from retention import classify_targets, summarize  # noqa: E402


def file_hash(p: Path):
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True, choices=["dev", "eval"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--mode", choices=["gt", "pred"], default="gt",
                    help="gt: 제공 트랙 실험(순위가 GT uid), pred: 실제 검출 실험")
    ap.add_argument("--representations", nargs="*", default=["object", "scene"],
                    help="인덱스에 포함된 표현 (E05: scene, E06: object)")
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    full = load_eval_config(args.config)
    cfg = full["evaluation"]
    ds = full["datasets"][args.dataset]
    paths = {k: ROOT / v for k, v in ds.items() if isinstance(v, str)}
    run = Path(args.run_dir)
    if not run.is_absolute():
        run = ROOT / run

    manifest = {m["video_id"]: m for m in read_jsonl(paths["manifest"])}
    split_videos = {v for v, m in manifest.items() if m["split"] == args.split}
    gt_rows = [r for r in read_jsonl(paths["gt_tracks"]) if r["video_id"] in split_videos]
    queries = [q for q in read_jsonl(paths["queries"]) if q["split"] == args.split]
    qrels = [r for r in read_jsonl(paths["qrels"]) if r["video_id"] in split_videos]
    targets = [t for t in read_jsonl(paths["targets"]) if t["split"] == args.split]
    if not targets:
        write_json(run / "metrics.json", {"status": "no_targets", "metrics": None,
                                          "reason": f"{args.dataset}/{args.split} 평가 대상 0"})
        return 0

    # ---------------- pred ↔ GT
    pred_to_gt = None
    link_summary = None
    if args.mode == "pred":
        pred_rows = [r for r in read_jsonl(run / "tracks.jsonl") if r["video_id"] in split_videos]
        frames = read_jsonl(run / "frames.jsonl")
        grid = None
        if frames:
            grid = defaultdict(list)
            for f in frames:
                if f["video_id"] in split_videos and f.get("detector_status", "ran") == "ran":
                    grid[f["video_id"]].append(f["timestamp_ms"])
            grid = {v: sorted(set(ts)) for v, ts in grid.items()}
        links = link_tracks(pred_rows, gt_rows, cfg["iou_threshold"], cfg.get("time_tolerance_ms", 100),
                            cfg["min_match_frames"], cfg["min_match_ratio"], grid)
        write_jsonl(run / "track_links.jsonl", links)
        pred_to_gt = {l["pred_track_uid"]: l["gt_track_uid"] for l in links if l["linked"]}
        link_summary = {"pred_tracks": len(links), "linked": len(pred_to_gt),
                        "gt_tracks_covered": len(set(pred_to_gt.values()))}

    # ---------------- 검색 지표
    preds_raw = read_jsonl(run / "predictions.jsonl")
    predictions = {}
    reps = set()
    for p in preds_raw:
        res = sorted(p.get("results", []), key=lambda r: r["rank"])
        predictions[p["query_id"]] = [r.get("track_uid") for r in res]
        reps.add(p.get("representation"))
    ks = tuple(cfg.get("recall_k", [5, 10]))
    retrieval, per_query = (None, [])
    if preds_raw:
        retrieval, per_query = evaluate_rankings(queries, qrels, predictions, pred_to_gt, ks)

    # ---------------- 보존·누락
    samples = [s for s in read_jsonl(run / "samples.jsonl") if s["video_id"] in split_videos]
    items_path = run / "items.jsonl"
    encoded = {i["item_id"] for i in read_jsonl(items_path)} if items_path.is_file() else None
    retention = None
    tres = []
    if samples:
        tres = classify_targets(targets, samples, gt_rows, manifest, cfg, encoded, pred_to_gt,
                                tuple(args.representations))
        retention = summarize(tres)
        retention["encoding_checked"] = encoded is not None
        write_jsonl(run / "eval_targets.jsonl", tres)

    # retrieval_missing: indexed 정답이 상위 K(최대 K)에 없음
    retrieval_missing = None
    if per_query and tres:
        kmax = max(ks)
        status = {r["target_id"]: r for r in tres}
        rel_by_q = defaultdict(list)
        for r in qrels:
            rel_by_q[r["query_id"]].append(r)
        cnt = defaultdict(int)
        for q in per_query:
            ranking = predictions.get(q["query_id"]) or []
            if pred_to_gt is not None:
                from metrics import collapse_ranking
                ranking = collapse_ranking(ranking, pred_to_gt)
            top = set(ranking[:kmax])
            for r in rel_by_q[q["query_id"]]:
                st = status.get(r["target_id"])
                if st is None or st["status"] == "N/A":
                    cnt["n/a"] += 1
                elif st["status"] != "indexed":
                    cnt[st["reason"]] += 1
                elif r["gt_track_uid"] in top:
                    cnt["retrieved"] += 1
                else:
                    cnt["retrieval_missing"] += 1
        retrieval_missing = {"k": kmax, "query_target_pairs": dict(cnt)}

    write_jsonl(run / "eval_per_query.jsonl", per_query)
    out = {
        "status": "success",
        "dataset": args.dataset, "split": args.split, "mode": args.mode,
        "evaluation_version": full.get("evaluation_version"),
        "config": cfg,
        "inputs": {k: file_hash(p) for k, p in paths.items()} | {
            "predictions": file_hash(run / "predictions.jsonl"), "samples": file_hash(run / "samples.jsonl"),
            "items": file_hash(items_path), "pred_tracks": file_hash(run / "tracks.jsonl")},
        "counts": {"videos": len(split_videos), "queries": len(queries),
                   "queries_with_qrels": len({r["query_id"] for r in qrels}),
                   "targets": len(targets), "samples": len(samples),
                   "prediction_representations": sorted(r for r in reps if r)},
        "track_links": link_summary,
        "retrieval": retrieval,
        "retention": retention,
        "retrieval_missing": retrieval_missing,
    }
    write_json(run / "metrics.json", out)
    print(f"[evaluate] {args.dataset}/{args.split} mode={args.mode} -> {run / 'metrics.json'}")
    if retrieval:
        a = retrieval["all"]
        print(f"  R@5={a['recall@5']:.4f} R@10={a['recall@10']:.4f} MRR={a['mrr']:.4f} (n={a['n_queries']})")
    if retention:
        tr = retention["track"]
        print(f"  track retention={tr['retention_rate']} missing={tr['missing_by_reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
