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
python -m pytest modules/b_detection_tracking/tests/test_litedetr_adapter.py -v
```

B1 프레임 읽기:

```bash
python modules/b_detection_tracking/src/validate_b1_fixtures.py \
  --project modules/b_detection_tracking \
  --config configs/vision.yaml \
  --output /tmp/cctv_b1_fixture_validation.json
```

## 5. 주요 파일

- `src/b_video_io.py`: PTS 기반 영상 프레임 읽기
- `src/litedetr_adapter.py`: 사람 탐지 결과 변환
- `src/run_litedetr_bytetrack_cpu.py`: ByteTrack 공통 처리
- `src/run_t3_multivideo.py`: 다중 영상 T3 추적
- `src/build_b3_b4_t3_multivideo.py`: B3/B4 JSONL 생성
- `configs/vision.yaml`: T3 기본 설정

## 6. C/F 연동

기본 연동 데이터는 별도 전달한 `baseline_T3` 결과를 사용한다.

원본 CCTV 영상, 모델 체크포인트, 대량 실행 결과 JSONL은 GitHub 저장소에 포함하지 않는다.

## 7. 제한 사항 및 후속 작업

- 실제 Lite-DETR 모델 추론은 별도로 수행한 Colab 코드가 필요하다.
- 공용 저장소에서 원본 영상부터 전체 파이프라인을 재실행하는 검증은 아직 완료되지 않았다.
- A/F와 서로 다른 영상 3개의 박스 수동 검수는 완료 여부 확인이 필요하다.
- G341 v2 확대 추론은 탐지 성능이 개선됐으나 중복 추적 후보가 증가하여 실험 버전으로 유지한다.
