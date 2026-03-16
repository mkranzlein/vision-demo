"""Faster R-CNN vehicle detector."""

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 9  # 8 vehicle classes + background


def create_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Create a Faster R-CNN model with a custom classification head.

    Args:
        num_classes: Number of output classes (including background).
        pretrained: If True, use COCO-pretrained weights.

    Returns:
        A Faster R-CNN model ready for fine-tuning.
    """
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
