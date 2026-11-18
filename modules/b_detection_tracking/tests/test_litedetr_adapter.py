import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from litedetr_adapter import convert_detections


def test_convert_detections():
    sample = {
        "scores": np.array([0.95, 0.80, 0.40, 0.10]),
        "labels": np.array([3, 6, 1, 8]),
        "boxes": np.array([
            [100, 200, 400, 500],  # car: 제외
            [500, 300, 900, 700],  # bus: 제외
            [100, 100, 200, 200],  # person: 포함
            [50, 50, 100, 100],    # truck: 제외
        ], dtype=float),
    }

    detections = convert_detections(
        sample,
        width=1920,
        height=1080,
        confidence_threshold=0.25,
    )

    assert len(detections) == 1

    person = detections[0]

    assert person["class_id"] == 1
    assert person["class_name"] == "person"
    assert person["confidence"] == 0.40
    assert person["bbox"] == [100.0, 100.0, 200.0, 200.0]


def test_person_below_threshold():
    sample = {
        "scores": np.array([0.10]),
        "labels": np.array([1]),
        "boxes": np.array([[10, 20, 100, 200]], dtype=float),
    }

    detections = convert_detections(
        sample,
        width=1920,
        height=1080,
        confidence_threshold=0.25,
    )

    assert detections == []


def test_person_bbox_clamping():
    sample = {
        "scores": np.array([0.90]),
        "labels": np.array([1]),
        "boxes": np.array([[-10, -20, 2000, 1100]], dtype=float),
    }

    detections = convert_detections(
        sample,
        width=1920,
        height=1080,
        confidence_threshold=0.25,
    )

    assert len(detections) == 1
    assert detections[0]["bbox"] == [0.0, 0.0, 1920.0, 1080.0]
