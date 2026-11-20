# CCTV 검색 팀 통합 저장소

A–G 팀원 전체 코드를 모아 병합하는 공용 저장소입니다. 이 통합본은 B의 CCTV 프레임 리더·추적 산출물에 C의 SigLIP2 인덱스와 자연어 검색을 연결합니다. A의 `samples.jsonl`도 입력으로 받을 수 있습니다.

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

B 산출물을 직접 쓰면 성공적으로 디코딩한 프레임은 전체 장면으로, 추적 관측은 객체로 인덱싱합니다. `--frames`만 주면 전체 장면, `--tracks`만 주면 객체를 검색할 수 있습니다. GitHub에서 확인한 브랜치와 PR 범위에는 A의 샘플링 코드가 없어 이 경로를 추가했습니다.

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
