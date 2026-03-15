"""FastAPI application for vision demo."""

from importlib.metadata import version

from fastapi import FastAPI, UploadFile

from vision_demo.models import DetectionResponse

app = FastAPI(title="Vision Demo", version=version("vision-demo"))


@app.get("/health")
async def health():
    """Return service health and version."""
    return {"status": "ok", "version": app.version}


@app.post("/detect", response_model=DetectionResponse)
async def detect(image: UploadFile):
    """Run object detection on an uploaded image."""
    raise NotImplementedError
