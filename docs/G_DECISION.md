# 결정 기록

| ID | 날짜 | 결정 | 기존 → 변경 | 이유 | 영향 | 재실행 범위 | 확인자 |
|---|---|---|---|---|---|---|---|
| G-001 | 2026-11-24 | AVI 프레임 시각은 fps 공식(프레임 번호×1000/fps) | PTS → fps_formula_avi_cfr | B의 PTS 시각이 일부 프레임에서 100ms 어긋나 FRAME_MISMATCH 발생 | C 임베딩 | 영상 단위 | |
| G-002 | 2026-11-24 | 작은 입력(smoke)은 A와 F만 연결 | 명세 G3(전 모듈) → A·F | Git에 영상이 없어 C·D·E 단계 실행 불가 | smoke | C·D·E는 G421 10초 run으로 대체 | |
| G-003 | 2026-11-24 | predictions는 run당 표현 1개, scene은 같은 시각 ±100ms 트랙으로 풀어 순위화 | D의 RRF 결합 → 단일 순위 | 평가기는 query_id당 1행만 읽고 scene(track_uid=null)은 무시함 | 검색 지표 | 모든 pred 평가 | |
| G-004 | 2026-11-24 | 정책 파일은 json으로 읽음 | yaml 읽기 → json | PyYAML은 1e-09를 문자열로 읽음 | A 설정 | 없음 | |