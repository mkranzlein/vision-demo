"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from vision_demo.models import Detection, DetectionResponse


def test_detection_valid():
    """Detection model accepts valid data."""
    d = Detection(label="cat", confidence=0.95, bbox=[10.0, 20.0, 100.0, 200.0])
    assert d.label == "cat"
    assert d.confidence == 0.95
    assert len(d.bbox) == 4


def test_detection_missing_field():
    """Detection model rejects missing fields."""
    with pytest.raises(ValidationError):
        Detection(label="cat", confidence=0.95)


def test_detection_response():
    """DetectionResponse holds a list of detections."""
    resp = DetectionResponse(
        detections=[
            Detection(label="cat", confidence=0.95, bbox=[10.0, 20.0, 100.0, 200.0]),
            Detection(label="dog", confidence=0.80, bbox=[50.0, 60.0, 150.0, 250.0]),
        ]
    )
    assert len(resp.detections) == 2


def test_detection_response_empty():
    """DetectionResponse accepts empty detections list."""
    resp = DetectionResponse(detections=[])
    assert resp.detections == []
