// UI 설정. 실제 D의 API를 연결할 때는 mock을 false로, apiBase를 API 주소로 바꾼다.
window.UI_CONFIG = {
  mock: true,                         // true: mock/ 폴더의 예시 응답 사용
  apiBase: "http://127.0.0.1:8000",   // D의 FastAPI 주소 (명세: 127.0.0.1, worker 1)
  seek_tolerance_ms: 250,             // 명세 E 2: seek 허용 오차
  top_k_default: 10,                  // 명세 D 4
  top_k_max: 50,
  text_max_len: 500,                  // 명세 D 2: 공백 처리 후 1~500자
  image_max_bytes: 5000000,
  image_max_pixels: 16000000,
  image_min_side_px: 32,
  mock_delay_ms: 400,                 // 가짜 API의 응답 지연
  mock_video_url: ""   // mock 모드에서 쓸 로컬 영상(없으면 화면에서 파일 선택)
};
