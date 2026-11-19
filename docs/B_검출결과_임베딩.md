# B tracks.jsonl → 이미지 임베딩 입력

사용자가 제공한 `B_LiteDETR_T3_Handoff.zip`의 실제 출력에 맞춘 입력 처리기다.
`embedding/track_input.py`가 전처리를 담당하고 `tools/embed_tracks.py`가 모델 호출과 분할 저장을 연결한다.
기존 `embedding build`, A의 샘플 선택 정책, D의 검색 코드는 변경하지 않는다.

## 입력과 처리

```text
tracks.jsonl 또는 tracks.jsonl.gz, 한 줄에 검출 객체 하나
  → 스키마·ID·유한 수치 검증
  → 원본 동영상에서 frames.jsonl이 가리키는 프레임 순차 디코딩
  → frame_index·실제 PTS·크기·source_offset_ms 일치 확인
  → 원본 픽셀 xyxy bbox crop, 기본 여백 10%
  → PIL RGB 이미지 배치
  → 기존 encoder.encode_image(images)
  → float32 [N, encoder.dim], L2 정규화 검사
```

검출 JSON 자체나 bbox 숫자를 임베딩하지 않는다. 원본 프레임의 객체 이미지를 임베딩한다.
리사이즈·정규화·텐서 변환은 기존 CLIP/SigLIP2 encoder의 모델 전처리를 사용한다.
원본 bbox는 보존하고, 실제 정수 crop 좌표는 `crop_bbox`에 별도로 기록한다.

ZIP 실측 입력은 c007 2,154건, c033 21,329건, c034 24,680건으로 총 48,163건이다.
ZIP의 `quality`에는 `area_ratio`, `blur_var`, `border_touch`만 있다. 처음 제시된 예시의 `score`는 선택 항목이며,
없어도 처리하고 임의로 계산하지 않는다. confidence와 quality는 출력 메타데이터에 그대로 남긴다.

기본적으로 모든 유효한 검출을 처리한다. `--min-confidence 0.25`처럼 지정한 경우에만 그 값 미만을 제외한다.
흐림·크기·클래스 필터와 트랙당 대표 프레임 선택은 추가하지 않았다. 한 트랙의 서로 다른 프레임은 각각 임베딩한다.

## 바로 연결하는 Python API

프로젝트 루트에서 실행하는 코드다. `records`는 파일뿐 아니라 dict를 하나씩 반환하는 수신 iterator도 가능하다.

```python
from embedding import create_encoder, load_config, iter_track_batches
from embedding.track_input import iter_jsonl

encoder = create_encoder(load_config("configs/embedding.siglip2.yaml"))

# B에서 configure_reader()로 설정한 리더를 연결한다.
# read_frame(video_id, timestamp_ms)는 아래 프레임 계약을 충족해야 한다.
for items, images in iter_track_batches(
    iter_jsonl("tracks.jsonl"), read_frame, batch_size=4, margin=0.10
):
    vectors = encoder.encode_image(images)
    # items[i]가 vectors[i]의 원본 추적 메타데이터다.
    # 이 지점에서 DB/벡터 저장소로 배치 단위로 전달한다.
```

모델 반환값 검사까지 포함하려면 다음 API를 사용한다.

```python
from embedding import iter_embedding_batches

for items, vectors in iter_embedding_batches(
    iter_jsonl("tracks.jsonl"), read_frame, encoder, batch_size=4
):
    # vectors: float32 [len(items), encoder.dim], 단위 벡터
    pass  # 실제 저장소의 upsert 호출로 교체
```

프레임 리더 반환 계약:

```python
{
    "rgb": image_rgb_uint8,  # numpy uint8 [H,W,3], RGB 순서. image_rgb 키도 허용.
    "actual_timestamp_ms": 200,
    "frame_index": 2,
    "width": 1920,
    "height": 1080,
    "source_offset_ms": 0,
    "time_source": "pts",
}
```

같은 프레임의 연속 검출들은 배치 경계를 넘어도 한 번 읽은 프레임을 재사용한다.
그 외 프레임 캐시는 한 장만 유지한다. 외부 B 리더의 원래 함수는 프레임 호출마다 영상을 처음부터 읽으므로,
대량 처리에는 아래 CLI가 사용하는 `SequentialVideoReader`를 권장한다.

## ZIP에 담긴 B 결과로 실행

**원본 CCTV 동영상은 ZIP에 없다.** ZIP의 검수 JPEG는 일부 프레임이고 박스가 그려져 있으므로 임베딩 원본으로 쓰지 않는다.
각 카메라의 원본 `vdo.avi`가 있어야 전체 실행이 가능하다. manifest의 SHA256과 실제 영상 파일을 실행 전에 비교한다.
Lite-DETR/ByteTrack 재추론과 해당 가중치는 필요 없다.

기존 임베딩 환경에 PyAV만 추가한다. 새 환경이면 먼저 `requirements-siglip2.txt` 또는 `requirements-embedding.txt`를 설치한다.

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements-track-input.txt

# 아래 두 경로를 실제 ZIP 압축 해제 폴더와 원본 영상 위치로 지정한다.
$bExportRoot = "C:\컴퓨터비전\B_LiteDETR_T3_Handoff"
$bOriginalVideo = "C:\데이터\c007\vdo.avi"

.\.venv311\Scripts\python.exe -m tools.embed_tracks `
  --tracks "$bExportRoot\outputs\c007\tracks.jsonl" `
  --frames "$bExportRoot\outputs\c007\frames.jsonl" `
  --manifest "$bExportRoot\support\data\manifest.jsonl" `
  --video $bOriginalVideo `
  --video-id cityflow_S02_c007 `
  --config configs/embedding.siglip2.yaml `
  --output runs/c007-track-embeddings
```

c033/c034는 해당 `outputs` 경로, 원본 영상, video ID, 출력 폴더를 각각 바꿔 별도로 실행한다.
video ID는 `cityflow_S05_c033`, `cityflow_S05_c034`다. dev/eval 구분은 manifest에서 보존한다.
CLI는 한 번에 영상 하나를 처리하며, `tracks.jsonl`과 `frames.jsonl`은 원본 프레임 순서로 정렬되어 있어야 한다.
정렬을 위해 전체 입력을 메모리에 올리지 않으며, 역순 요청은 오류로 중단한다.

기본 모델 배치는 기존 config를 따른다. 현재 SigLIP2 설정은 GPU 4, CPU 2다.
`--batch-size 4`로 변경할 수 있다. 모델 OOM은 즉시 오류로 보고하므로 메모리에 맞춰 배치를 줄여 새 출력 폴더로 재실행한다.

## 분할 결과

```text
runs/c007-track-embeddings/
  part-000000/
    items.jsonl
    vectors.npy
  part-000001/
    items.jsonl
    vectors.npy
  manifest.json
```

작은 모델 배치마다 파일을 만들지 않고 메타데이터·벡터를 기본 4,096건 단위로 모아 저장한다.
`--shard-size`로 조정할 수 있다. 분할은 모델 배치 경계에서 이루어지므로 마지막을 제외한 분할 크기는
`shard_size` 이상, `shard_size + batch_size - 1` 이하다. crop 이미지는 저장하지 않는다.
메모리는 전체 검출 수가 아닌, 한 프레임·모델 배치 crop·한 분할의 메타데이터/벡터 크기에 비례한다.
호출자가 generator를 `list(...)`로 바꾸거나 누적 저장하면 이 보장은 사라진다.

각 분할에서 `items.jsonl`의 `vector_row`가 같은 폴더의 `vectors.npy` 행을 가리킨다.
`item_id`는 video/frame/실제 시각/track_uid로 정하며, UID는 B가 준 값을 그대로 보존한다.
중복 제거는 하지 않는다. 같은 관측이 중복 전달되면 같은 item_id로 반복 반환되므로 수신 시스템에서 중복 정책을 정한다.
`manifest.json`에 모델 ID·리비전·전처리 해시·차원·정규화·원본 영상 해시와 총 행 수를 기록한다.

분할 폴더는 두 파일 작성 후 공개한다. 전체 성공 시에만 `manifest.json`을 작성한다.
오류 시 `failure.json`에 이미 저장한 행 수와 오류를 남기고 종료 코드 1로 중단한다.
잘못된 JSON·좌표·프레임·모델 벡터를 조용히 건너뛰지 않는다. 자동 재시작/재개는 구현하지 않았다.
소비자는 성공한 `manifest.json`이 있는 결과만 읽어야 한다. 빈 입력도 0건/0분할로 정상 처리한다.

이 분할 출력은 새 입력 파이프라인의 결과이며 기존 D의 단일 `index_manifest.json` 인덱스 형식과 다르다.
기존 `validate_index()`에 그대로 전달할 수 없으며, D 연동 시 분할별로 읽거나 별도 인덱스 변환이 필요하다.
B ZIP의 `det01`은 과거 YOLO 실행과 충돌할 수 있다. 서로 다른 검출 실행은 출력/저장 공간을 분리해야 한다.

## 확인

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p test_track_input.py -v
.\.venv311\Scripts\python.exe -m unittest discover -s tests -v
```

입력 검증, 선택적 quality.score, B/C 이미지 키, RGB 순서, 배치의 지연 소비, 프레임 재사용,
소수 bbox crop, 시각 불일치, 순차 디코딩과 offset, 벡터 행 대응, 분할 저장과 실패 공개 방지를 검사한다.
원본 CCTV 영상이 없으므로 실제 3개 영상 전체 디코딩·모델 처리량·검색 정확도는 확인 대상에 포함되지 않는다.

순차 디코딩은 [PyAV 공식 예제](https://pyav.org/docs/stable/cookbook/basics.html)의 `container.decode(stream)`을 사용한다.
