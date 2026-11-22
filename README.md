# CCTV 검색 팀 통합 저장소

**11월 21일 통합 평가**에서 MEVA 평가 영상 2개의 전체 시간대를 4초 간격으로 검색했을 때 한국어·영어 모두 상위 5개에 정답 장면이 든 질의는 6/12였습니다. D의 인물 추적 API는 같은 영상의 첫 10초에서 정답 사람 박스까지 3/12를 찾았습니다. [질의 유형별 결과와 재현 방법](evaluation/reports/2026-11-21-meva/README.md)

## 전체 영상 검색

프레임 상태 가설을 시험하려면 `run-search.cmd events --dataset meva --split eval --question "체육관에서 두 사람이 포옹하는 장면"`을 실행하세요. 원본에서 활동 영역과 2~4초 구간을 선정하고, 검색 결과에 시작·종료 시각을 표시합니다. [기존 B→A와의 정확도 비교 및 재현 방법](evaluation/reports/2026-11-22-frame-state/README.md)

기본 검색은 F 목록의 MEVA 6개와 VIRAT 11개를 읽습니다. 영상마다 B의 200ms 사람 관측에 A의 `configs/person/hybrid.yaml`을 실행한 뒤, C가 A의 `samples.jsonl`에 있는 장면과 사람 crop만 임베딩합니다. 질문은 dev·eval 양쪽의 완성된 인덱스에서 검색하며 영상 전체를 통틀어 상위 K개를 반환합니다. 평가 점수 산출에는 기존 분할별 평가 명령을 사용합니다.

Python 3.11 환경에서 PyTorch와 torchvision을 설치하고 `python -m pip install -r requirements-full.txt`를 실행하세요. CUDA용 PyTorch를 설치하면 GPU를 사용할 수 있습니다. Windows 기본 실행은 다음과 같습니다.

```powershell
.\run-search.cmd
.\run-live.cmd
```

첫 명령은 터미널 검색입니다. 웹 검색은 두 번째 명령을 별도로 실행하고 준비가 끝나면 [웹 화면](http://127.0.0.1:8780/ui/)을 엽니다. 기본 범위는 **전체 영상 · A 선별**입니다. 카메라 하나로 좁히려면 영상 ID를 입력하세요. `gallery_dev`는 사용자가 직접 등록한 이벤트 저장소입니다.

최초 실행은 [원본 목록과 사용 조건](data/README.md)에 기재된 원본을 내려받아 SHA-256을 확인합니다. B 인계가 없는 영상은 Lite-DETR와 ByteTrack T3부터 처리하므로 CPU에서는 여러 시간이 걸릴 수 있습니다. 터미널에 영상별 진행률이 표시됩니다. B 관측과 완성된 인덱스는 다음 실행에서 재사용합니다. 원본·모델·생성된 인덱스는 로컬에 저장합니다.

기본 경로는 원본 `runs/source-data`, 기존 B 출력 `runs/b-inputs`, 생성 캐시 `runs/a-full`입니다. 원본이 다른 곳에 있다면 두 실행기 모두 환경 변수 `CCTV_DATA_ROOT`를 사용합니다. 기존 B 결과는 `CCTV_B_INPUTS` 아래 영상별 폴더에 `frames.jsonl`, `tracks.jsonl`, B의 `validation_report.json`을 두세요. manifest와 완료 상태 파일을 갖춘 인계도 지원합니다. 첫 10초만 있는 부분 인계는 건너뛰고 해당 원본의 전체 B 관측을 생성합니다.

영상과 주석의 출처는 MEVA KF1(Kitware·IARPA, CC BY 4.0) 및 VIRAT Ground 2.0입니다. VIRAT 영상에는 VIRAT Video Data Set Protection Agreement가 적용됩니다. [MEVA 출처](data/meva/data_inventory.csv) · [VIRAT 출처](data/virat/data_inventory.csv)

### 상위 파이프라인에서 이벤트 보내기

완료된 MP4 한 구간을 HTTP 본문에 담아 보냅니다. 한 구간의 상한은 120초, 128 MiB, 18,000프레임, 1,600만 화소입니다. 응답 `202`는 접수이며 `GET /api/status`의 `jobs[].state`가 `ready`가 되면 검색할 수 있습니다. 동일한 파일은 SHA-256으로 중복을 제거합니다.

```powershell
curl.exe --data-binary "@segment.mp4" -H "Content-Type: video/mp4" "http://127.0.0.1:8780/api/events/clip?name=segment.mp4"
```

B 담당 출력의 경우 `runs/live/inbox/<batch>/` 안에 dev manifest.jsonl, frames.jsonl, tracks.jsonl, video_status.jsonl과 manifest가 가리키는 상대 경로의 영상을 모두 둡니다. query_exclusions.jsonl은 선택입니다. 쓰기가 끝나면 고유 배치 이름으로 `POST /api/events/handoff/<batch>`를 전송하고 완료될 때까지 입력을 보존하세요. 같은 이름은 같은 배치로 취급합니다. `GET /docs`에서 API를 확인할 수 있습니다.

이 서비스는 이 PC의 127.0.0.1에서 실행합니다. 단일 워커와 디스크 대기열은 접수 순서대로 작업합니다. 대기·처리 중 작업 32개부터 새 요청은 HTTP 429로 응답하며 재전송할 수 있습니다. 재시작하면 미완료 작업을 재개하고 완료 인덱스를 복원합니다. 웹 상태 목록은 최근 32건이고 검색 범위는 완료된 전체 dev 인덱스입니다. 검색은 인덱스별 정확 코사인 비교이므로 대규모 처리량과 지연은 별도 부하 시험이 필요합니다.

### 제출 포지션 확인

| 담당 | 범위 | 통합 근거 |
|---|---|---|
| A · sungchoo | 이벤트 판정과 샘플링 | PR #3 · `2de8736`, 6대 관측 manifest 보완 `2880474` |
| B · SEI135 | 영상 읽기, Lite-DETR·ByteTrack | PR #1 · `b6bcfe7` |
| C · wellseekogi | SigLIP2 임베딩, 벡터, 로컬 Qwen | 제출 브랜치 `1a68008` |
| D · JangSungWook | 검색 순위·API·영상 제공 | PR #5 · `923d555`, 실제 C 인덱스 및 SigLIP2 연동 |
| E · seonjin217 | 검색 UI·박스 표시·영상 재생 | PR #4 · `eb3154b` |
| F · JunHan-K | VIRAT·MEVA 데이터와 평가 | PR #2 · `5da44de` |
| G | 통합 실행·실험·납품 | 별도 제출 브랜치 미확인. 이 통합본에서 로컬 실행·검증 연결 |

2026-11-21 기준 제출 내역입니다. C의 원래 모듈 안내는 [C 모듈 문서](docs/C_MODULE.md)에 있습니다.

## 한 번만 질의하기

```powershell
.\run-search.cmd full --question "병원 복도에서 걸어가는 사람" --top-k 5
```

결과에는 영상 ID, 프레임 번호, 시각, 유사도와 이미지 경로가 표시됩니다. 4초 카페 예제를 따로 확인할 때만 `run-search.cmd demo`를 사용하세요.

## 팀 산출물로 질의하기

원본 영상, `cctv-3.0` `manifest.jsonl`, A의 `samples.jsonl`이나 B가 내보낸 `frames.jsonl`·`tracks.jsonl`을 준비하세요. `--data-root`는 manifest의 `relative_path`를 해석하는 원본 영상 기준 폴더입니다. 입력이 참조하는 영상만 열어 파일 해시와 시각·프레임 인덱스를 대조합니다. B의 구형 T3 manifest처럼 `schema_version`만 생략된 경우도 읽을 수 있습니다. dev와 eval은 각각 인덱스를 만드세요.

A 샘플이 있으면 A의 선택 결과와 출처를 그대로 사용합니다.

```powershell
.\run-search.cmd build --manifest data/dev/manifest.jsonl --data-root data/videos --samples data/dev/samples.jsonl --output runs/a-dev
.\run-search.cmd ask --index runs/a-dev
```

B 산출물을 직접 쓰면 성공적으로 디코딩한 프레임은 전체 장면으로, 추적 관측은 객체로 인덱싱합니다. `--frames`만 주면 전체 장면, `--tracks`만 주면 객체를 검색할 수 있습니다.

```powershell
.\run-search.cmd build --manifest data/dev/manifest.jsonl --data-root data/videos --frames data/dev/frames.jsonl --tracks data/dev/tracks.jsonl --output runs/b-dev
.\run-search.cmd ask --index runs/b-dev --top-k 5 --question "사람이 걸어가는 장면을 찾아줘"
```

인덱스 생성 시 새 출력 경로를 지정하세요. `ask --representation scene`으로 장면을, `ask --representation object`로 객체를 골라 검색할 수 있습니다. 위 실행기는 SigLIP2 순위를 반환합니다. C의 Qwen 기반 안내 문장과 HTTP API는 별도 [프레임 추천 연동 안내](docs/D_프레임추천_연동.md)에 있습니다.

## 브랜치와 병합

- `main`: 통합 확인을 마친 안정 버전
- `develop`: 팀원 작업을 모으는 통합 브랜치
- `feature/<담당>-<작업>`: 담당자 작업 브랜치. 예: `feature/c-embedding`

작업 브랜치 → `develop` PR → 담당자 검토와 연동 확인 → `main` PR 순서로 병합합니다.
이 절차는 팀 운영 규칙이며, GitHub의 강제 브랜치 보호는 아직 설정하지 않았습니다.

## 팀원 시작

공개 저장소이므로 누구나 조회·복제할 수 있습니다. 원본 저장소에 직접 push하려면 관리자의 collaborator 초대를 수락해야 합니다. 초대 없이 기여하려면 fork에서 작업한 뒤 PR을 보내세요.

```powershell
git clone https://github.com/wellseekogi/cctv-search-team.git
cd cctv-search-team
git switch develop
git pull --ff-only origin develop
git switch -c feature/c-embedding
```

`c-embedding`은 자신의 담당과 작업 이름으로 바꿉니다. 기존 개인 프로젝트의 코드만 필요한 경로에 복사하고, `.git` 폴더는 복사하지 않습니다. 공통 파일 구조와 의존성 변경은 담당자와 맞춥니다.

```powershell
git status --short
git add <변경한-파일이나-폴더>
git diff --cached --check
git diff --cached
git commit -m "feat: 담당 기능 추가"
git push -u origin HEAD
```

GitHub에서 PR의 base를 `develop`로 선택하고, 변경 내용·실행 방법·검증 결과를 적습니다. 병합 전 최신 `develop`을 작업 브랜치에 반영하고 충돌을 해결한 뒤 관련 검증을 다시 실행하세요.

## 기존 C 담당 저장소

[wellseekogi/cctv-search](https://github.com/wellseekogi/cctv-search)는 C 담당 기존 코드와 실험 이력을 보관합니다. 이 통합본은 C의 `feature/qwen35-4b` 커밋 `1a68008`에서 임베딩과 RAG 모듈을 가져왔습니다.

## Git에 넣지 않는 파일

API 키·비밀번호·`.env`, 원본 CCTV 영상·개인정보, 모델 가중치, 가상환경, 캐시, 실행 결과는 커밋하지 않습니다.
`.gitignore`만으로 모든 민감 파일이 차단되지는 않으므로 커밋 전 staged diff를 직접 확인하세요.

## A 샘플링 및 B/F 연동

[A 실행·입력·검증 안내](docs/A_SAMPLING.md)를 확인하세요. B/F 자료는 외부 경로로 지정하며 대량 실행 결과는 Git에 포함하지 않습니다.
