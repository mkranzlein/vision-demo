"""FastAPI application for vision demo."""

import io
import logging
from importlib.metadata import version
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from PIL import Image
from torchvision.transforms import v2

from vision_demo.data.coco import CONTIGUOUS_TO_LABEL
from vision_demo.model.detector import create_model
from vision_demo.models import Detection, DetectionResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Vision Demo", version=version("vision-demo"))

# Model loaded once at startup
_model = None
_device = None

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "detector.pth"
CONFIDENCE_THRESHOLD = 0.5


def _get_model():
    """Load the model on first request."""
    global _model, _device  # noqa: PLW0603
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = create_model(pretrained=False)
        model_path = Path(DEFAULT_MODEL_PATH)
        if model_path.exists():
            _model.load_state_dict(torch.load(model_path, map_location=_device, weights_only=True))
            logger.info("Loaded model from %s", model_path)
        else:
            logger.warning("No checkpoint at %s, using untrained model.", model_path)
        _model.to(_device)
        _model.eval()
    return _model, _device


@app.get("/health")
async def health():
    """Return service health and version."""
    return {"status": "ok", "version": app.version}


@app.post("/detect", response_model=DetectionResponse)
async def detect(image: UploadFile):
    """Run object detection on an uploaded image."""
    model, device = _get_model()

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}") from e

    transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    img_tensor = transform(pil_image).to(device)

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    detections = []
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"], strict=True):
        if score.item() < CONFIDENCE_THRESHOLD:
            continue
        # label is 1-indexed (0=background), contiguous map is 0-indexed
        class_idx = label.item() - 1
        label_name = CONTIGUOUS_TO_LABEL.get(class_idx, f"class_{class_idx}")
        detections.append(
            Detection(
                label=label_name,
                confidence=round(score.item(), 4),
                bbox=[round(c, 2) for c in box.tolist()],
            )
        )

    return DetectionResponse(detections=detections)
