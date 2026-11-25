# evaluation/ — F 담당: 정답 매칭·검색 지표·보존 평가

실제 웹 채팅 115회와 MEVA 평가 영상 검색의 질의별 결과를 확인할 수 있습니다. [검색 정확도·반복 검증 보고서](reports/2026-11-20-meva/README.md)

데이터셋과 무관하게 동작합니다. CityFlow 담당은 `configs/evaluation/evaluation.yaml`의 `datasets:`에 항목만 추가하면 됩니다.
현재 등록된 데이터셋: `virat`, `meva` (예: `--dataset meva`). 모의 run은 `make_smoke_run.py --dataset meva ...`로 만듭니다.
설정: `configs/evaluation/evaluation.yaml` (`evaluation_version: eval_v1`)

## 실행
```
python evaluation/evaluate.py --dataset virat --split dev  --run-dir runs/<run_id> --mode gt
python evaluation/evaluate.py --dataset virat --split eval --run-dir runs/<run_id> --mode pred
python evaluation/evaluate.py ... --representations scene    # E05 (scene만)
python evaluation/evaluate.py ... --representations object   # E06 (object만)
python evaluation/tests/test_evaluation.py                   # 손계산 대조 12개
```
G의 `pipeline evaluate`는 `evaluation.evaluate.main([...])`를 호출하면 됩니다.

## 다른 담당자가 맞춰야 할 입력 형식 (run 폴더)
| 파일 | 작성 | 평가기가 읽는 필드 |
|---|---|---|
| `predictions.jsonl` | D | `{"query_id", "representation", "results": [{"rank", "track_uid", "score"}...]}` — 질의당 1행, **전체 순위**. scene 결과(`track_uid=null`)는 무시 |
| `samples.jsonl` | A | `item_id, video_id, timestamp_ms, representation(scene/object), track_uid, bbox` |
| `items.jsonl` | C | `item_id` (인코딩 성공 행만). 파일이 없으면 인코딩 검사 생략 |
| `tracks.jsonl` | B | pred 관측 (`--mode pred`): `video_id, track_uid, timestamp_ms, bbox` |
| `frames.jsonl` | B | 선택. `detector_status=ran` 시각을 GT 판정 가능 시각으로 사용 |

## 계산 규칙 (명세서 F 5·F 6)
- **pred↔GT 연결** (`matching.py`): 100ms 이내 GT 프레임, IoU≥0.5 쌍을 IoU 순(동률 ID 사전순) 프레임별 일대일 →
  pred 트랙별 최다 매칭 GT (동률: 평균 IoU → GT ID) → 매칭 ≥3프레임 & 판정 가능 관측 대비 ≥0.5면 연결 → `track_links.jsonl`
- **Recall@K / MRR** (`metrics.py`): 질의별 계산 후 평균. 분모는 정답이 있는 전체 질의(예측 누락 = 0점).
  pred 실험은 같은 GT의 조각을 첫 등장으로 합쳐 재순위, 미매칭 pred는 오답으로 유지
- **보존** (`retention.py`): object = 샘플 박스와 GT IoU≥0.5이고 CLIP 입력(여백 10% crop→짧은 변 224→center crop)에서
  보이는 GT 폭·높이 ≥8px. scene = center crop 후 GT 면적 ≥50% 남고 입력 크기 ≥8px.
  활동 = 3등분 각 부분에 필수 참여자 유효 관측 + 서로 다른 시점 ≥3
- **누락 분류**: detection_missing → sampling_missing → encoding_missing (대상 단위), retrieval_missing (질의×정답 단위, K=10)
- evaluable=false 대상은 N/A로 따로 집계, 평가 대상 0이면 `metrics=null`과 사유 기록

## 출력 (run 폴더)
`metrics.json` (검색·보존·연결 요약 + 입력 해시), `eval_per_query.jsonl`, `eval_targets.jsonl`, `track_links.jsonl`

## 검사 결과 (2026-11-09)
| 검사 | 입력 | 기대값 | 실측값 | 결과 |
|---|---|---|---|---|
| 명세서 손계산 (1위·4위·미검색) | 3질의 | R@5=2/3, MRR=0.4167 | 0.6667 / 0.4167 | PASS |
| 다중 정답 Recall | 3정답 중 2위·7위 | R@5=1/3, R@10=2/3, RR=0.5 | 일치 | PASS |
| 예측 누락 질의 | 2질의 중 1개 누락 | 분모 2, MRR=0.5 | 일치 | PASS |
| pred 조각 합치기·미매칭 유지 | 5개 순위 | G2가 4위 | 4위 | PASS |
| 연결 임계값 (매칭 2프레임) | – | 미연결 | 미연결 | PASS |
| 프레임별 일대일 | 같은 GT에 pred 2개 | IoU 높은 쪽만 | 일치 | PASS |
| 비율 규칙 (3/10) | – | 미연결 | 미연결 | PASS |
| scene center crop·크기 | 1920×1080 | 가장자리·20px 제외 | 일치 | PASS |
| object IoU·가시성 | – | IoU<0.5 제외 | 일치 | PASS |
| 보존 분류 (indexed/encoding/sampling/N/A) | 4대상 | 1/1/1/N/A, 누락률 2/3 | 일치 | PASS |
| 활동 3등분 | 200·400·2200ms | 중간 부분 누락 | sampling_missing | PASS |
| detection_missing (pred) | 미연결 GT | detection_missing | 일치 | PASS |
| 실제 VIRAT 전체 경로 (모의 run 3개) | `runs/f_smoke_*` | oracle 1.0, random≈0, 오류 없이 완료 | 일치 | PASS |

`make_smoke_run.py`는 **평가기 연결 검사용 모의 데이터**입니다. 실험 결과로 사용하지 않습니다.

## 팀 저장소에서 실행할 때 (2026-11-19)
- 필요한 패키지: PyYAML (`common.py`가 `configs/evaluation/evaluation.yaml`을 읽음)
- 손계산 테스트: `python evaluation/tests/test_evaluation.py` → 12개 PASS
- smoke 예제 대조: `fixtures/smoke/predictions.jsonl` + `qrels.jsonl`
  - `metrics.evaluate_rankings`로 계산하면 recall@5 0.3333, recall@10 0.8333, mrr 0.5625입니다.
  - `fixtures/smoke/expected_metrics.json`과 일치합니다.
- `evaluate.py`·`make_smoke_run.py`로 전체 데이터를 평가하려면 `data/<dataset>/tracks_gt.jsonl`과 `activity_labels.jsonl`이 필요합니다.
  - 두 파일은 Git에 없습니다(`data/README.md` 참조).
- 위 검사 결과 표의 모의 run(`runs/f_smoke_*`)은 사람만 탐지 기준 적용 전 qrels로 돌린 값입니다.
