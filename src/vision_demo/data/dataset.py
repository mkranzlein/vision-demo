"""PyTorch Dataset for COCO vehicle detection."""

import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2


class CocoVehicleDataset(torch.utils.data.Dataset):
    """Dataset that loads COCO vehicle images and bounding box annotations.

    Args:
        data_dir: Path to a split directory (e.g. data/coco/train/) containing
            an ``images/`` subdirectory and an ``annotations.json`` file.
        transforms: Optional torchvision v2 transforms to apply.
    """

    def __init__(self, data_dir: Path, transforms: v2.Compose | None = None) -> None:
        """Initialize dataset from a split directory."""
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.transforms = transforms

        with (self.data_dir / "annotations.json").open() as f:
            coco = json.load(f)

        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.image_id_to_idx = {img["id"]: idx for idx, img in enumerate(self.images)}

        self.anns_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_image.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        """Load an image and its target annotations.

        Args:
            idx: Index into the dataset.

        Returns:
            Tuple of (image, target) where target contains boxes, labels, and image_id.
        """
        img_info = self.images[idx]
        img_path = self.images_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")

        anns = self.anns_by_image.get(img_info["id"], [])

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                # +1 because 0 is background for Faster R-CNN
                labels.append(ann["category_id"] + 1)

        canvas_size = (img_info["height"], img_info["width"])
        if boxes:
            boxes_tensor = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=canvas_size)
        else:
            boxes_tensor = tv_tensors.BoundingBoxes(torch.zeros((0, 4)), format="XYXY", canvas_size=canvas_size)

        target = {
            "boxes": boxes_tensor,
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": img_info["id"],
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def get_transforms(train: bool = True) -> v2.Compose:
    """Get torchvision v2 transforms for training or evaluation.

    Args:
        train: If True, include data augmentation transforms.

    Returns:
        A Compose transform pipeline.
    """
    transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    if train:
        transforms.append(v2.RandomHorizontalFlip(0.5))
    return v2.Compose(transforms)
