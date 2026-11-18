# fixtures/smoke — 평가 입력 형식 예제

`data/virat/`에서 영상 1개(`VIRAT_S_040103_03_000284_000425`, dev)만 잘라 낸 작은 예제입니다. 전체 데이터 없이 파일 형식을 확인하거나 평가 코드를 빠르게 시험할 때 씁니다. 형식은 실제 데이터와 같습니다(`cctv-3.0`).

| 파일 | 행 | 내용 |
|---|---|---|
| `manifest.jsonl` | 1 | 영상 정보 |
| `tracks_gt.jsonl` | 1,272 | GT 트랙 11개의 전체 관측 |
| `targets.jsonl` | 11 | 평가 대상 |
| `queries.jsonl` | 2 | `vq_d04` a little girl in a pink outfit / `vq_d05` a man pulling a green cart |
| `qrels.jsonl` | 6 | 질의별 정답 |

트랙 11개는 세 종류입니다.
- **정답 트랙 6개:** vq_d04의 g001·g005·g006·g016·g043, vq_d05의 g014
- **방해 트랙 4개:** 평가 가능한 사람 트랙 g000·g002·g003·g007
- **지원하지 않는 클래스 1개:** car 트랙 g015. `evaluable=false`이고 `na_reason=unsupported_class`입니다.

vq_d05의 원래 정답이던 car 트랙(g031)은 사람만 탐지 기준에 따라 qrels에서 뺐습니다(`data/virat/excluded_queries.jsonl`).

영상 파일은 들어 있지 않습니다. 프레임이 필요하면 `data/README.md`의 방법으로 해당 영상을 받습니다.
