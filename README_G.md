# G 통합 실행기 (`pipeline`)

**실험용 통합 실행기입니다. B(검출·추적)는 실행하지 않습니다.**
B가 미리 만든 `frames.jsonl`·`tracks.jsonl`을 입력으로 받아, A 샘플링부터 C 임베딩, D 검색, E 웹 화면까지 한 번에 실행합니다.
영상만 넣어 B부터 처리하는 경로는 이 폴더에 없습니다. B 인계가 없는 영상을 처리하는 방법은 `integration/web-search-variants` 브랜치 README의 "전체 영상 검색"과 "웹에서 검색 버전 선택"을 참고하세요.

## 실행 범위

| 단계 | 담당 | 여기서 |
|---|---|---|
| manifest | F | 새로 만들지 않고 `data/meva/manifest.jsonl`의 행을 복사해 사용 혹은 make_handoff.py를 사용하여 임의 제작 가능 |
| 검출·추적 | B | **실행하지 않음.** 결과 파일을 입력으로 받음 |
| 샘플링 | A | `pipeline run` |
| 이미지·임베딩·인덱스 | C | `pipeline run` (SigLIP2) |
| 검색·웹 화면 | D·E | `pipeline serve` (통합 서버 `team_search.live`) |
| 평가 | F | `pipeline evaluate` (미검증, 정답 트랙 파일 필요) |

## 입력 폴더

`runs/live/inbox/<배치>/`에 아래 파일을 둡니다.

- `manifest.jsonl` — F의 manifest 행. `video_id`는 B의 `frames.jsonl` 값과 같아야 합니다.
- `frames.jsonl`, `tracks.jsonl` — B의 출력 (`cctv-3.0`)
- `video_status.jsonl` — 영상별 `video_id`, `available_end_ms`, `video_status`
- manifest의 `relative_path`가 가리키는 원본 영상

```powershell
python pipeline\make_handoff.py G421 <B 출력 폴더> <원본 AVI>
```

원본 영상 SHA-256이 manifest 값과 다르면 중단합니다. 영상은 Git에 없으며 `data/meva/data_inventory.csv`의 주소에서 받습니다.

## 실행

Python 3.11 환경에서 `requirements-live.txt`를 설치합니다. NVIDIA GPU를 쓰면 CUDA용 PyTorch를 설치하세요. 모든 명령은 저장소 루트에서 같은 Python으로 실행합니다.

```powershell
python -m pipeline doctor --device cuda
python -m pipeline validate --input runs\live\inbox\g421
python -m pipeline run --input runs\live\inbox\g421 --run-id g421_t10 --device cuda --start-ms 256000 --end-ms 266000
python -m pipeline serve --run-id g421_t10
```

- `--start-ms/--end-ms`는 `frames`·`tracks`를 시간으로 거릅니다. 영상 파일은 자르지 않습니다.
- 정책은 기본 `hybrid`이고 `--policy`로 바꿉니다.
- `run-id`는 새 이름이어야 합니다. 결과는 `runs/<run-id>/`에 쌓이고 Git에는 올라가지 않습니다.
- 웹 화면은 `http://127.0.0.1:8780/ui/`입니다.

## 시각 규칙

MEVA의 AVI는 PTS가 일부 프레임에서 100ms 어긋나 `FRAME_MISMATCH`가 났습니다. 그래서 manifest의 `time_source=fps_formula_avi_cfr`에 맞춰 프레임 번호×1000÷fps로 시각을 매깁니다(`pipeline/reader.py`). `docs/decisions.md` G-001을 보세요.

## 확인 상태 (2026-11-24, G421 256~266초 한 번)

| 항목 | 결과 |
|---|---|
| 입력 검사, 샘플링, 임베딩, 서버 등록 | 성공. 이벤트 89건, 벡터 104개 ([결과](docs/results/g421_t10)) |
| 웹 검색 | 질의에 따라 결과가 달라짐. 한 질의의 후보 104, 응답 6,635ms (코드상 Qwen 답변 생성 시간 포함) |
| 구간·bbox 표시 | 정보 패널에 구간 261000–263000 ms와 bbox 표시 확인 |
| **브라우저 영상 재생** | **실패.** 원본 AVI를 브라우저가 재생하지 못함(오류 코드 4). |
| `pipeline/make_playback.py`을 사용하여 구간 영상 임의 제작 가능 |
| 정확도(Recall·MRR) | 미측정. F의 `tracks_gt.jsonl`이 Git에 없음 |
| 다른 PC 재현 | 미확인 |

## 한계

- 한 번에 영상 하나·구간 하나를 다루며, 정답이 구간 밖에 있으면 검색에 나오지 않습니다.
- D의 `api/` 서버와 응답 형식이 다릅니다. 이슈 #14에서 정리 중입니다.
- 지금 `--store`로 한 run만 띄웁니다. 여러 run을 한 화면에서 보거나 검색 버전을 비교하는 기능은 `integration/web-search-variants`에 있습니다.