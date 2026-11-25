"""Lite-DETR PostProcess 결과를 CCTV 객체 탐지 형식으로 변환."""

import numpy as np

COCO_TARGET_CLASSES = {
    1: "person"
}


def convert_detections(
    result,
    width: int,
    height: int,
    confidence_threshold: float = 0.25,
):
    """
    Lite-DETR 공식 PostProcess의 이미지 1장 결과를 변환한다.

    result = {
        "scores": Tensor[N],
        "labels": Tensor[N],
        "boxes": Tensor[N, 4]  # 원본 픽셀 기준 xyxy
    }

    반환:
    [
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,
            "class_name": str,
            "class_id": int
        }
    ]
    """

    def to_numpy(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    scores = to_numpy(result["scores"]).reshape(-1)
    labels = to_numpy(result["labels"]).reshape(-1)
    boxes = to_numpy(result["boxes"])

    if boxes.shape != (len(scores), 4):
        raise ValueError("boxes shape must be (N, 4)")

    if len(scores) != len(labels):
        raise ValueError("scores and labels length mismatch")

    detections = []

    for score, label, box in zip(scores, labels, boxes):
        score = float(score)
        class_id = int(label)

        if not np.isfinite(score) or not np.all(np.isfinite(box)):
            continue

        if not 0 <= score <= 1 or score < confidence_threshold:
            continue

        if class_id not in COCO_TARGET_CLASSES:
            continue

        x1, y1, x2, y2 = map(float, box)

        bbox = [
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
        ]

        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            continue

        detections.append({
            "bbox": bbox,
            "confidence": score,
            "class_name": COCO_TARGET_CLASSES[class_id],
            "class_id": class_id,
        })

    return detections
