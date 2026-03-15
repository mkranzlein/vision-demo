"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from vision_demo.app import app

client = TestClient(app)


def test_detect_not_implemented():
    """Detect endpoint raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        client.post("/detect")
