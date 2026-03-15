"""FastAPI application for vision demo."""

from importlib.metadata import version

from fastapi import FastAPI

app = FastAPI(title="Vision Demo", version=version("vision-demo"))


@app.post("/detect")
async def detect():
    """Run object detection on an uploaded image."""
    raise NotImplementedError
