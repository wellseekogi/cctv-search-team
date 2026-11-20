# CCTV 검색 팀 통합 저장소

## 실시간 웹 채팅

Windows에서 `run-live.cmd`를 실행한 뒤 [로컬 웹 화면](http://127.0.0.1:8780/ui/)을 엽니다. **예제 영상 등록**을 누르고 **검색 가능**이 뜨면 “빨간 자동차가 나오는 장면을 찾아줘”라고 보내세요. Qwen3.5-4B가 검색된 프레임 번호와 시각을 받아 답변합니다. 후보를 누르면 E의 플레이어가 해당 시각으로 이동합니다. 대화의 최근 네 메시지가 다음 질문에 전달됩니다. 프레임 검색 자체는 현재 질문을 사용합니다.

예제는 6초 합성 영상과 정답 박스를 사용합니다. 직접 MP4 구간을 올리면 A의 motion 정책이 첫 프레임과 밝기 변화 기준을 넘은 장면을 선별합니다. B 추적 결과가 있는 인계 묶음은 A의 hybrid 정책으로 객체 이벤트와 주기 장면을 함께 선별합니다. 원본 영상만 올리는 경로에서 Lite-DETR를 실행하지 않습니다. 모델이 답변을 생성하지 못하면 화면에 오류가 표시됩니다.

새 PC에서는 아래 CLI 설치 절차에 따라 Python 3.11과 PyTorch를 준비하고 `python -m pip install -r requirements-live.txt`를 실행하세요. 첫 실행 시 SigLIP2와 Qwen 가중치, Windows CPU용 llama.cpp를 내려받아 해시를 검사합니다. 이 PC에서는 인접한 C 저장소의 환경과 모델을 재사용합니다.

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
| A · sungchoo | 이벤트 판정과 샘플링 | PR #3 · `2de8736` |
| B · SEI135 | 영상 읽기, Lite-DETR·ByteTrack | PR #1 · `b6bcfe7` |
| C · wellseekogi | SigLIP2 임베딩, 벡터, 로컬 Qwen | 제출 브랜치 `1a68008` |
| D | 검색 순위·API·영상 제공 | 별도 제출 브랜치 미확인. 이 통합본에서 웹 검색·미디어 API 연결 |
| E · seonjin217 | 검색 UI·박스 표시·영상 재생 | PR #4 · `eb3154b` |
| F · JunHan-K | VIRAT·MEVA 데이터와 평가 | PR #2 · `5da44de` |
| G | 통합 실행·실험·납품 | 별도 제출 브랜치 미확인. 이 통합본에서 로컬 실행·검증 연결 |

2026-11-20 기준 원격 브랜치와 전체 PR을 7인 실행명세서와 대조했습니다. 계정명은 제출 기록 기준입니다. C의 원래 모듈 안내는 [C 모듈 문서](docs/C_MODULE.md)에 있습니다.

## 바로 질의하기

이 PC에서 [run-search.cmd](run-search.cmd)를 열거나 이 폴더의 PowerShell에서 실행하세요.

```powershell
.\run-search.cmd
```

`질문 >`에 `빨간 자동차` 또는 `파란 자전거`를 입력하세요. 결과는 영상 ID, 프레임 번호, 재생 시각, 코사인 유사도와 프레임 이미지 경로를 보여 줍니다. `종료` 또는 `/exit`로 끝냅니다. 첫 실행에서 6초 합성 영상과 인덱스를 만들고, 이후에는 `runs/team-demo`를 재사용합니다. 데모의 박스와 트랙 ID는 생성한 테스트 데이터입니다. 실제 Lite-DETR 추론은 B의 [실행 안내](modules/b_detection_tracking/README.md)를 따릅니다.

실행기는 현재 폴더의 `.venv`, 옆 `cctv-search` 폴더의 `.venv311`, 시스템 Python 3.11 순으로 사용합니다. 같은 모델이 옆 폴더에 있으면 기존 가중치를 재사용합니다. 새 PC에서는 Python 3.11로 환경을 한 번 설치하세요.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-siglip2.txt -r requirements-track-input.txt
.\run-search.cmd
```

모델이 없는 환경에서는 첫 실행에 SigLIP2 가중치 약 1.5GB를 내려받아 해시를 검사합니다. 실행기에 `demo --question "빨간 자동차"`를 넘기면 한 번 질의하고 끝납니다.

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
