# 로컬 테스트 전용: 자연어 이미지 검색과 LLM 답변

이 폴더는 개발자가 직접 확인하는 도구입니다. 제품용 API와 D 클라이언트는 별도로 실행합니다. 테스트 사진·벡터·실행 기록은 Git에서 제외된 `runs/traffic30-real`에 둡니다.

VS Code에서 `C:\컴퓨터비전\cctv-search-team` 폴더를 열고 터미널에서 실행하세요.

```powershell
cd C:\컴퓨터비전\cctv-search-team
.\.venv-siglip2\Scripts\python.exe -X utf8 -m devtools.terminal_search
```

SigLIP2와 Qwen3.5-4B를 한 번 로딩한 뒤 `질문 >`가 나오면 원하는 자연어 문장을 직접 입력합니다.

```text
질문 > 버스가 있는 사진을 찾아줘
```

입력한 질문을 실제로 임베딩하고, 사진 벡터와 비교해 상위 원본 사진 3장을 검색합니다. 검색한 사진의 원본 파일명 3개와 질문을 Qwen에 전달합니다. **LLM이 파일명 3개를 포함한 한국어 추천 문장을 직접 생성하며, 터미널에는 `답변 >`와 생성 문장을 표시합니다.** 벡터·점수·이미지 미리보기는 출력하지 않습니다.

다음 질문을 계속 입력할 수 있고, `/exit` 또는 `종료`로 끝냅니다. 별도 서버를 켤 필요 없이 이 명령이 테스트용 LLM을 실행하고 종료 시 정리합니다. 기본 내부 포트는 `18766`이며 충돌하면 `--llm-port 18767`을 붙입니다. 테스트용 런타임 로그는 `runs/traffic30-real/terminal-llm`에 저장합니다.

데이터가 없는 새 로컬 환경에서만 먼저 실행합니다.

```powershell
.\.venv-siglip2\Scripts\python.exe -X utf8 -m devtools.traffic_demo
```

현재 데이터는 실제 교통 사진 30장입니다. 원본 전체 이미지 30개의 벡터를 검색하며, 객체 crop은 이 테스트의 검색 대상에 포함하지 않습니다. `sources.json`에서 원본 사진 이름을 가져옵니다. Qwen 가중치와 런타임은 기존 로컬 설정을 재사용하며, 없는 경우 고정된 파일을 내려받고 해시를 검사합니다.

생성 문장에 검색한 파일명들이 정확히 한 번씩 순서대로 포함되어 있는지 검사합니다. 생성이 실패하거나 잘못된 파일명을 출력하면 오류를 표시하고 다음 질문을 받습니다. 고정 문구를 LLM 답변처럼 대신 출력하지 않습니다.

사진은 독립된 이미지로, CCTV 영상의 실제 시간·추적 데이터가 아닙니다. 검색 유사도는 정답 확률이 아니며 검색 정확도는 별도 평가가 필요합니다.

인터프리터 오류가 있으면 `Ctrl+Shift+P` → `Python: Select Interpreter` → `Enter interpreter path`에서 `C:\컴퓨터비전\cctv-search-team\.venv-siglip2\Scripts\python.exe`를 선택하세요.
