"""Tests for the FastAPI application."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from vision_demo.app import app

client = TestClient(app)


def test_health():
    """Health endpoint returns ok status and version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def _make_test_image() -> io.BytesIO:
    """Create a minimal valid PNG image in memory."""
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_detect_returns_detections():
    """Detect endpoint returns a DetectionResponse."""
    image = _make_test_image()
    response = client.post("/detect", files={"image": ("test.png", image, "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_detect_invalid_image():
    """Detect endpoint returns 400 for invalid image data."""
    dummy = io.BytesIO(b"not an image")
    response = client.post("/detect", files={"image": ("bad.png", dummy, "image/png")})
    assert response.status_code == 400


def test_detect_detection_format():
    """Each detection has label, confidence, and bbox fields."""
    image = _make_test_image()
    response = client.post("/detect", files={"image": ("test.png", image, "image/png")})
    data = response.json()
    for det in data["detections"]:
        assert "label" in det
        assert "confidence" in det
        assert "bbox" in det
        assert len(det["bbox"]) == 4
        assert 0.0 <= det["confidence"] <= 1.0
