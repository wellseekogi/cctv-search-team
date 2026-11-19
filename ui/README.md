# ui/ — CCTV 검색 화면 (담당 E)

D의 API 예시 JSON을 가짜 API로 써서 먼저 만든 화면입니다. 빌드 도구 없이 HTML·CSS·JS만 사용합니다.

## 실행

```bash
python3 serve.py 8765        # 저장소 루트의 serve.py (Range 요청 지원)
# 브라우저에서 http://127.0.0.1:8765/
```

`python3 -m http.server`는 Range 요청을 지원하지 않아 영상 seek가 동작하지 않습니다. 사용하지 마세요.
실제 D 서버를 쓸 때는 FastAPI가 `ui/`를 정적 파일로 함께 서비스하면 됩니다.

## 설정 (`config.js`)

| 키 | 의미 |
|---|---|
| `mock` | true면 `mock/` 예시 응답 사용. D의 API 연결 시 false |
| `apiBase` | D의 API 주소 (명세: 127.0.0.1) |
| `seek_tolerance_ms` | seek 허용 오차 250 (명세 E 2) |
| `mock_video_url` | mock 모드의 영상. 비워 두면 화면의 "로컬 영상 선택"으로 파일을 고름 |

## mock 모드 입력어

검색어에 아래 단어가 들어가면 해당 상태를 보여 줍니다. 그 외에는 D의 예시 응답을 돌려줍니다.

`empty` 결과 없음 · `404` UNKNOWN_VIDEO · `422` INVALID_INPUT · `503` INDEX_NOT_READY · `slow` 3초 지연

## 화면이 지키는 규칙

- 입력 검사: 검색어 공백 제거 후 1~500자, top_k 1~50, 이미지 JPEG/PNG·5,000,000 바이트 이하·각 변 32px 이상·1,600만 화소 이하. `video_ids`를 비우면 필드를 보내지 않음(전체 gallery).
- 요청 번호로 가장 최근 요청의 응답만 화면에 반영.
- 결과 선택 시 `loadedmetadata` 후 `currentTime = (representative_ms − source_offset_ms)/1000`. 실제 seek 오차를 화면에 PASS/FAIL로 표시.
- 박스: `scale = min(viewW/origW, viewH/origH)` 와 `object-fit: contain` 여백(offset)으로 원본 bbox를 변환. 창 크기가 바뀌면 다시 그림.
- "구간 재생": `start_ms`부터 재생해 `end_ms`에서 정지.

## 점검 기록 (자동 점검, 1280×800 / 420×800)

- 화면 박스 좌표가 계산식 값과 일치 (두 크기 모두).
- 빈 결과·404·422·503·빈 입력·501자·느린 응답 뒤 새 요청 시 오래된 응답 무시 확인.
- 아직 안 한 것: 사람 2명의 직접 점검(명세), 손상 이미지 파일 점검, 실제 D 서버 연결, 데모 녹화.
