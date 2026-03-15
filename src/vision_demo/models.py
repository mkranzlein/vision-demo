"""Pydantic models for detection request and response."""

from pydantic import BaseModel


class Detection(BaseModel):
    """A single detected object."""

    label: str
    confidence: float
    bbox: list[float]
    """Bounding box as [x_min, y_min, x_max, y_max]."""


class DetectionResponse(BaseModel):
    """Response from the /detect endpoint."""

    detections: list[Detection]
