"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from vision_demo.app import app

client = TestClient(app)


def test_health():
    """Health endpoint returns ok status and version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_detect_not_implemented():
    """Detect endpoint raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        client.post("/detect")
