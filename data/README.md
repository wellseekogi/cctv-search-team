# data/ — VIRAT·MEVA 평가 데이터 (F 담당)

스키마 `cctv-3.0` · 질의 버전 `virat_queries_v1`, `meva_queries_v1` · 갱신 2026-11-19

이 폴더에는 평가에 필요한 **작은 메타데이터만** 들어 있습니다. 영상과 GT 관측(`tracks_gt.jsonl`, VIRAT 60MB·MEVA 48MB)은 Git에 넣지 않으므로 아래 방법으로 각자 받습니다.

## 1. 사람만 탐지 기준 (2026-11-18 팀 결정)
- `configs/class_map.yaml`의 `supported_classes`는 `[person]`입니다.
- `targets.jsonl`에서 car·vehicle·bicycle·object 트랙은 `evaluable=false`, `na_reason=unsupported_class`입니다. GT 자체는 지우지 않았습니다.
- `qrels.jsonl`:
  - **차량만 정답인 질의는 뺐습니다.** VIRAT에서 19개(text 14, image 5)를 뺐고, MEVA에서는 뺀 질의가 없습니다.
  - **사람과 탈것이 섞인 질의는 남기고, 사람 트랙만 정답으로 둡니다.** VIRAT vq_d05·vq_d09·vq_di03, MEVA mq_d07·mq_e04가 여기에 해당합니다.
- 뺀 질의와 qrels에서 뺀 트랙은 `excluded_queries.jsonl`에 사유와 함께 기록되어 있습니다.
- VIRAT 텍스트 질의는 dev 8, eval 7입니다. 명세의 "각 10개 이상"에 못 미치지만, VIRAT은 보조 데이터라서 이대로 둡니다.

## 2. 들어 있는 파일
| 파일 | VIRAT | MEVA | 내용 |
|---|---|---|---|
| `manifest.jsonl` | 11 | 6 | 영상별 크기·길이·fps·split·SHA256. `relative_path`는 `data_root` 기준 |
| `targets.jsonl` | 890 | 3,470 | 평가 대상(track·activity)과 `evaluable`, `na_reason` |
| `queries.jsonl` | 20 (text 15 + image 5) | 37 (text 27 + image 10) | 영어 질의 |
| `qrels.jsonl` | 36 | 83 | 질의별 정답 트랙 |
| `excluded_queries.jsonl` | 22 | 2 | 사람만 탐지 기준으로 뺀 질의·트랙 |
| `data_inventory.csv` | 11 | 6 | 출처 URL·릴리스·주석 버전·SHA256·크기·라이선스 |
| `split_summary.json` | – | – | 분할별 그룹·영상·길이·평가 가능 트랙·질의 수 |

공통 설정은 `configs/evaluation.yaml`(평가 기준값·데이터 경로)과 `configs/class_map.yaml`(클래스 대응)입니다. 작은 예제는 `fixtures/smoke/`에 있습니다.

**Git에 없는 파일:** `tracks_gt.jsonl`, `activity_labels.jsonl`, `query_images/*.png`, `image_query_exclusions.jsonl`
- `configs/evaluation.yaml`에 경로가 적혀 있지만 이 저장소에는 없습니다. 영상과 주석을 받은 뒤 F의 변환 스크립트로 만들거나 F에게 받습니다.
- 질의 이미지는 영상에서 잘라 낸 것이라 넣지 않았습니다. VIRAT은 재배포 조건이 있습니다.

## 3. 원본 데이터 받는 방법
먼저 `configs/local.example.yaml`을 `configs/local.yaml`로 복사하고 `data_root`를 자기 경로로 바꿉니다. 아래 경로는 모두 `data_root` 기준입니다.

### VIRAT Release 2.0 Ground (영상 11개, 25.03분)
1. https://viratdata.org 의 **VIRAT Video Data Set Protection Agreement** 조건을 확인합니다. 영상은 재배포하지 않습니다.
2. 주석을 받아 `virat/annotations/`에 풉니다.
   - 주소: `https://data.kitware.com/api/v1/item/67914e81f07d3010406a7d21/download`
   - 파일: `annotations-and-docs.zip`
3. 영상을 `virat/videos/<video_id>.mp4`로 받습니다.
   - `manifest.jsonl`의 11개 `video_id`가 대상입니다.
   - 주소: `https://data.kitware.com/api/v1/item/<item_id>/download`
   - 영상별 다운로드 URL과 해시는 `data/virat/data_inventory.csv`에 있습니다.
4. 받은 파일의 SHA256을 `data_inventory.csv`의 `file_sha256`과 비교합니다.

### MEVA KF1 (영상 6개, 30.01분, CC BY 4.0)
1. 영상을 받아 `meva/videos/`에 둡니다.
   - 주소: `data_inventory.csv`의 `source_url` (`https://mevadata-public-01.s3.amazonaws.com/drops-123-r13/...r13.avi`)
   - 공개 버킷이라 계정이 필요 없습니다. 연결이 느리면 Range 요청으로 나눠 받습니다.
2. 주석을 받습니다.
   - 저장소: https://gitlab.kitware.com/meva/meva-data-repo 의 `annotation/DIVA-phase-2/MEVA/kitware` (커밋 f844579a)
   - 클립별 `*.geom.yml`, `*.types.yml`, `*.activities.yml`을 `meva/annotations/`에 둡니다.
   - 예: `git clone --filter=blob:none --sparse https://gitlab.kitware.com/meva/meva-data-repo.git`
3. 받은 파일의 SHA256을 `data_inventory.csv`의 `file_sha256`과 비교합니다.

## 4. 핵심 규칙
- 시각은 원본 기준 정수 ms입니다. 박스는 `[x1, y1, x2, y2]`이고 프레임 번호는 0부터 셉니다.
- `track_uid` 형식은 두 가지입니다.
  - VIRAT: `gtv1:{video_id}:g###`
  - MEVA: `gtm1:{video_id}:m####`
- `target_id`는 `t:{track_uid}` 형식입니다.
- 격자 관측이 3개 이하인 짧은 트랙은 평가하지 않습니다(`na_reason=fewer_than_4_grid_observations`).
- 분할은 seed 42로 나눴습니다.
  - VIRAT: 장면 단위, eval 33.6%
  - MEVA: 촬영 위치 단위로 실내·실외를 층화, eval 33.3%
- 이미지 질의는 참조 시점 전후 1초를 모든 정책의 인덱스 후보에서 뺍니다.
