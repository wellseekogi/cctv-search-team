# B 담당: Lite-DETR + ByteTrack

## 1. 담당 기능

- B1: CCTV 영상 읽기 및 200ms 주기 프레임 샘플링
- B2: PTS 기반 실제 시각, 프레임 인덱스, 원본 좌표 반환
- B3: `cctv-3.0` 형식 관측 JSONL 및 품질값 생성
- B4: `vision.yaml` 설정, 검증 및 오류 기록

## 2. 모델 및 설정

- 탐지 모델: Lite-DETR
- 탐지 대상: 사람(person, COCO class_id=1)
- 추적 알고리즘: ByteTrack
- 기본 전달 버전: T3
- 모델 변경: YOLO11n에서 Lite-DETR로 변경(팀 합의)

실제 모델 및 추적 설정은 `configs/vision.yaml`을 참조한다.

## 3. 검증 결과

- Lite-DETR 사람 탐지 어댑터: pytest 3개 통과
- B1 합성 영상 프레임 검증: 12개 통과
- MEVA 6개 영상 처리: 9,004프레임
- T3 추적 관측값: 50,206건
- 출력 오류 레코드: 0건

## 4. 테스트 실행

사람 탐지 어댑터:

```bash
python -m pytest src/pipeline/b_detection_tracking/tests/test_litedetr_adapter.py -v
```

B1 프레임 읽기:

```bash
python src/pipeline/b_detection_tracking/src/validate_b1_fixtures.py \
  --project src/pipeline/b_detection_tracking \
  --config configs/vision.yaml \
  --output /tmp/cctv_b1_fixture_validation.json
```

## 5. 주요 파일

- `src/b_video_io.py`: PTS 기반 영상 프레임 읽기
- `src/litedetr_adapter.py`: 사람 탐지 결과 변환
- `src/run_litedetr_bytetrack_cpu.py`: ByteTrack 공통 처리
- `src/run_t3_multivideo.py`: 기존 다중 영상 T3 추적
- `src/build_b3_b4_t3_multivideo.py`: 기존 다중 영상 B3/B4 JSONL 생성
- `notebooks/litedetr_cctv_generic_inference.ipynb`: 범용 Lite-DETR 사람 탐지 Colab 노트북
- `src/run_generic_bytetrack_t3.py`: 범용 ByteTrack T3 추적
- `src/build_b3_b4_generic_person.py`: 범용 B3/B4 JSONL 생성 및 품질값 계산
- `configs/vision.yaml`: T3 기본 설정

## 6. C/F 연동

기본 연동 데이터는 별도 전달한 `baseline_T3` 결과를 사용한다.

원본 CCTV 영상, 모델 체크포인트, 대량 실행 결과 JSONL은 GitHub 저장소에 포함하지 않는다.

## 7. 제한 사항 및 후속 작업

- 실제 Lite-DETR 추론은 `notebooks/litedetr_cctv_generic_inference.ipynb`에서 수행하며, GPU 환경과 모델 체크포인트가 필요하다.
- G330 영상의 범용 파이프라인은 전체 실행 및 데이터 정합성 검증을 완료했으나, 공용 저장소를 새 환경에서 처음부터 재실행하는 검증은 아직 완료되지 않았다.
- G330 시각화 이미지 10장의 수동 검수는 아직 완료되지 않았다.
- A/F 담당과 서로 다른 CCTV 영상 3개의 박스 수동 검수 및 통합 검증은 아직 완료 여부 확인이 필요하다.
- ByteTrack의 고유 Track ID 수는 실제 등장 인원수를 의미하지 않으며, ID 유지 성능은 별도 평가가 필요하다.
- G341 v2 확대 추론은 탐지 성능이 개선됐으나 중복 추적 후보가 증가하여 실험 버전으로 유지한다.

## 8. 범용 Lite-DETR + ByteTrack 파이프라인

### 8.1 처리 흐름

CCTV 영상 → Lite-DETR 사람 탐지 → ByteTrack T3 추적 → B3/B4 JSONL 생성 → 결과 검증

기존 MEVA 6개 영상의 T3 결과는 유지하면서, 새로운 CCTV 영상에도 적용할 수 있도록 범용 추론 및 추적·변환 코드를 별도로 구성하였다.

### 8.2 실행 순서

1. Google Colab에서 `notebooks/litedetr_cctv_generic_inference.ipynb`를 실행한다.
2. Google Drive의 `data/videos` 폴더에서 입력 영상을 선택한다. 필요한 경우 S3 Key를 지정해 MEVA 영상을 다운로드할 수 있다.
3. `SMOKE_TEST = True`로 일부 프레임을 시험한 뒤, 전체 실행 시 `SMOKE_TEST = False`로 설정한다.
4. Lite-DETR이 생성한 `detections_<video_id>_pts.jsonl` 파일을 `run_generic_bytetrack_t3.py`에 입력한다.
5. 탐지 JSONL, 추적 JSONL, 원본 영상을 `build_b3_b4_generic_person.py`에 입력해 최종 B3/B4 결과를 생성한다.
6. `validation_report.json`과 `validation_visuals`를 확인한다.

### 8.3 G330 전체 영상 검증 결과

검증 영상: `2018-03-10.10-25-00.10-30-00.school.G330.r13.avi`

| 항목 | 결과 |
|---|---:|
| 프레임 샘플링 주기 | 200ms |
| 처리 프레임 | 1,502 |
| Lite-DETR 탐지 후보 | 103,965 |
| ByteTrack 추적 관측 | 21,198 |
| 고유 Track ID | 735 |
| BBox 경계 보정 | 265 |
| 오류 레코드 | 0 |
| B3/B4 데이터 정합성 검사 | 12/12 통과 |

### 8.4 최종 출력 파일

- `frames.jsonl`: 프레임 시각, 크기 및 처리 상태
- `tracks.jsonl`: 추적 ID, 사람 bbox, confidence 및 품질값
- `errors.jsonl`: 처리 오류 기록
- `bbox_corrections.jsonl`: 영상 경계를 벗어난 bbox의 보정 기록
- `vision.yaml`: 실행 설정
- `validation_report.json`: 결과 수량 및 검증 정보
- `validation_visuals/`: 추적 결과 시각화 이미지

출력 형식은 `cctv-3.0`을 사용하며, Lite-DETR의 `track_source_id`는 `litedetr01`로 구분한다.

### 8.5 검증 범위

G330에서 전체 탐지·추적·B3/B4 생성 및 구조적 정합성 검사를 완료하였다.

다만 다음 항목은 추가 검증이 필요하다.

- 실제 영상과 bbox의 시각적 일치 여부
- 동일 인물의 Track ID 유지 성능
- 다른 영상에 대한 전체 파이프라인 재현성
- A/F 담당 모듈과의 통합 호환성