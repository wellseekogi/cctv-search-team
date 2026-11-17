# CCTV 검색 팀 통합 저장소

A–G 팀원 전체 코드를 모아 병합하는 공용 저장소입니다. 현재는 협업 안내와 Git 설정만 있으며, 담당자 코드와 실행 환경은 PR로 합칩니다.

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

[wellseekogi/cctv-search](https://github.com/wellseekogi/cctv-search)는 C 담당 기존 코드와 실험 이력을 보관합니다. 이 저장소로 자동 병합하지 않았습니다. 필요한 버전을 확인한 뒤 작업 브랜치에서 가져오세요.

## Git에 넣지 않는 파일

API 키·비밀번호·`.env`, 원본 CCTV 영상·개인정보, 모델 가중치, 가상환경, 캐시, 실행 결과는 커밋하지 않습니다.
`.gitignore`만으로 모든 민감 파일이 차단되지는 않으므로 커밋 전 staged diff를 직접 확인하세요.
설치와 통합 실행 명령은 실제 모듈 및 공통 의존성 병합 후 추가합니다.
