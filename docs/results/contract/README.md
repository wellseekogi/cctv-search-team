# 검색 응답 계약 · D 서버 검증

[이슈 #14](https://github.com/wellseekogi/cctv-search-team/issues/14)의 D 담당 자료입니다. 아래 네 예시는 develop의 D 앱에 HTTP 요청을 보내 캡처했습니다. 각 JSON에서 `request`는 재현 입력, `status`는 HTTP 상태, `response`는 실제 본문입니다.

| 사례 | 요청 | 확인할 결과 |
|---|---|---|
| [텍스트 검색](d-text.json) | POST /search, a white car, top_k=1 | 200. 후보 트랙 5개 중 1위와 별도 장면 목록 |
| [이미지 검색](d-image.json) | POST /search/image, multipart의 image 필드에 red_patch.png | 200. 동일한 검색 결과 구조 |
| [빈 결과](d-empty.json) | POST /search, video_ids=[] | 200. results=[], total_candidates=0 |
| [잘못된 입력](d-error.json) | POST /search, top_k=0 | 422. request_id와 error.code=INVALID_INPUT, error.message |

## E가 읽는 필드와 결과 단위

| 필드 | D 응답의 의미 |
|---|---|
| request_id, index_version, latency_ms | 요청 식별자, 검색 인덱스 버전, D 검색 처리 시간(ms). request_id는 X-Request-ID 헤더와 같습니다. |
| results[], total_candidates | 트랙당 한 행. total_candidates는 top_k 적용 전 트랙 수입니다. scene_results[]는 장면 순위이며 results[]와 별개입니다. |
| rank, video_id, score, score_type | 순위는 1부터 시작합니다. combined 트랙 순위의 score는 RRF, object·scene 단독은 cosine입니다. combined에서 별도 scene_results의 score는 장면 유사도입니다. |
| result_type, item_representation, track_uid | D 트랙 행은 result_type=track입니다. 대표 항목은 object 또는 scene일 수 있습니다. 최상위 representation은 요청 모드입니다. 통합 앱은 샘플링된 **프레임**당 한 행으로, 같은 트랙이 여러 번 나올 수 있습니다. |
| representative_ms, start_ms, end_ms | 원본 타임라인의 대표 시각과 재생 시작·종료 시각(ms). D는 기본 ±1초를 사용 가능 구간에 맞춥니다. E는 응답값을 그대로 사용합니다. 이벤트 종류나 행동 지속시간을 나타내지 않습니다. |
| source_offset_ms | 제공 파일의 0초에 대응하는 원본 시각(ms). 플레이어 시각(초) = (원본 시각 − source_offset_ms) / 1000. |
| bbox, width, height | 대표 시각 박스 한 개 [x1,y1,x2,y2], 기준 전체 프레임의 픽셀 크기. 전체 프레임 표시 영역에 비례해 그립니다. bbox=null이면 박스를 생략합니다. D의 별도 장면 행은 bbox와 track_uid가 null입니다. |
| thumbnail_url, crop_url, video_url | 서버 기준 미디어 경로. 박스는 전체 프레임 thumbnail_url에 표시합니다. 해당 URL이 null이면 해당 미디어 동작을 생략합니다. |
| answer, llm_status | 통합 앱 텍스트 채팅 확장입니다. D에서는 필드 없이 검색 목록과 플레이어가 동작해야 합니다. |

**재생 검산.** 텍스트 예시에서 파일 offset은 0이므로 결과를 선택하면 1.2초, 구간 재생은 0.2~2.2초입니다. [실제 G421 발췌 검증](https://github.com/wellseekogi/cctv-search-team/pull/13)에서는 offset이 258400ms입니다. 대표 시각 260400ms는 파일의 2초, 원본 259400~261400ms는 파일의 1~3초입니다. 이 발췌본의 좌표 기준은 960×536입니다.

## 재현

저장소 루트에서 다음을 실행합니다.

```powershell
python -m pip install -r requirements.txt
python -m api --config configs/retrieval/retrieval.yaml
```

별도 터미널에서 아래 명령으로 네 요청을 다시 보낼 수 있습니다. 요청 ID와 소요 시간은 실행마다 달라집니다.

```powershell
$base = 'http://127.0.0.1:8000'
foreach ($name in 'text', 'empty', 'error') {
    $sample = Get-Content "docs/results/contract/d-$name.json" -Raw -Encoding utf8 | ConvertFrom-Json
    $sample.request.json | ConvertTo-Json -Compress | curl.exe -sS -i -H 'Content-Type: application/json' --data-binary '@-' "$base/search"
}
curl.exe -sS -i -F 'image=@fixtures/d_small/queries/red_patch.png' -F 'gallery_id=gallery_dev' -F 'top_k=1' -F 'representation=combined' "$base/search/image"
```

## 통합 앱과의 합의 상태

[G 확인 사항](https://github.com/wellseekogi/cctv-search-team/issues/14#issuecomment-7139371989)은 프레임 단위 결과, scene의 null 박스, 대표 박스 한 건, 서버가 지정한 재생 구간입니다. D 트랙 순위는 평가에서도 사용하므로 프레임 단위로 해석하면 안 됩니다. 화면에서 행 단위를 구분하는 방법과 행별 representation 필수화, 통합 오류 형식, Qwen 실패 시 검색 결과 유지, 통합 앱 수정 담당자는 #14에서 결정할 항목입니다. D는 이미 구조화된 오류와 대표 박스·구간을 제공합니다.
