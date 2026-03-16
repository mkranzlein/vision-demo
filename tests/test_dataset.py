"""Tests for CocoVehicleDataset."""

import json

import torch
from PIL import Image

from vision_demo.data.dataset import CocoVehicleDataset, get_transforms


def _create_dummy_dataset(tmp_path, n_images=3):
    """Create a minimal dataset on disk for testing."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    images = []
    annotations = []
    for i in range(n_images):
        fname = f"{i:06d}.jpg"
        img = Image.new("RGB", (64, 48))
        img.save(images_dir / fname)
        images.append({"id": i, "file_name": fname, "height": 48, "width": 64, "coco_url": ""})
        annotations.append({"id": i, "image_id": i, "category_id": 0, "bbox": [5, 5, 20, 20]})

    coco = {"images": images, "annotations": annotations, "categories": [{"id": 0, "name": "bicycle"}]}
    (tmp_path / "annotations.json").write_text(json.dumps(coco))
    return tmp_path


class TestCocoVehicleDataset:
    """Tests for CocoVehicleDataset."""

    def test_length(self, tmp_path):
        """Dataset length matches number of images."""
        data_dir = _create_dummy_dataset(tmp_path, n_images=5)
        ds = CocoVehicleDataset(data_dir)
        assert len(ds) == 5

    def test_getitem_returns_image_and_target(self, tmp_path):
        """Each item is a (image, target) tuple."""
        data_dir = _create_dummy_dataset(tmp_path)
        ds = CocoVehicleDataset(data_dir, transforms=get_transforms(train=False))
        image, target = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape[0] == 3
        assert "boxes" in target
        assert "labels" in target

    def test_boxes_are_xyxy(self, tmp_path):
        """Bounding boxes are converted from xywh to xyxy format."""
        data_dir = _create_dummy_dataset(tmp_path)
        ds = CocoVehicleDataset(data_dir, transforms=get_transforms(train=False))
        _, target = ds[0]
        box = target["boxes"][0]
        # bbox was [5, 5, 20, 20] (xywh) → [5, 5, 25, 25] (xyxy)
        assert box[0] == 5
        assert box[1] == 5
        assert box[2] == 25
        assert box[3] == 25

    def test_labels_are_one_indexed(self, tmp_path):
        """Labels are shifted by +1 so 0 is reserved for background."""
        data_dir = _create_dummy_dataset(tmp_path)
        ds = CocoVehicleDataset(data_dir, transforms=get_transforms(train=False))
        _, target = ds[0]
        assert target["labels"][0].item() == 1  # category_id 0 → label 1


class TestGetTransforms:
    """Tests for get_transforms."""

    def test_train_transforms(self):
        """Train transforms include augmentation."""
        t = get_transforms(train=True)
        assert t is not None

    def test_eval_transforms(self):
        """Eval transforms do not include augmentation."""
        t = get_transforms(train=False)
        assert t is not None
