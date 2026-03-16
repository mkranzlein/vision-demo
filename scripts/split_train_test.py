"""Split the downloaded COCO vehicle training data into train and test sets.

Reads the filtered train annotations, splits by image ID, moves test images
to a separate directory, and writes new annotation files for each split.

Usage:
    python scripts/split_train_test.py
    python scripts/split_train_test.py --test-ratio 0.15 --seed 42
"""

import argparse
import json
import logging
from pathlib import Path

from vision_demo.data.coco import save_filtered_annotations, split_dataset, split_image_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "coco"


def main():
    """Split train data into train and test sets."""
    parser = argparse.ArgumentParser(description="Split COCO vehicle train data into train/test.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Root directory for COCO data.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Fraction of images for test set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    train_ann_path = data_dir / "train" / "annotations.json"

    if not train_ann_path.exists():
        logger.error("Train annotations not found at %s. Run download_coco.py first.", train_ann_path)
        return

    # Load filtered train annotations
    logger.info("Loading train annotations from %s...", train_ann_path)
    with train_ann_path.open() as f:
        train_coco = json.load(f)

    logger.info(
        "Original train set: %d images, %d annotations.",
        len(train_coco["images"]),
        len(train_coco["annotations"]),
    )

    # Split
    logger.info("Splitting with test_ratio=%.2f, seed=%d...", args.test_ratio, args.seed)
    train_split, test_split = split_dataset(train_coco, test_ratio=args.test_ratio, seed=args.seed)

    logger.info("Train split: %d images, %d annotations.", len(train_split["images"]), len(train_split["annotations"]))
    logger.info("Test split: %d images, %d annotations.", len(test_split["images"]), len(test_split["annotations"]))

    # Move test images
    train_images_dir = data_dir / "train" / "images"
    test_images_dir = data_dir / "test" / "images"
    split_image_files(train_split, test_split, train_images_dir, test_images_dir)

    # Save updated annotations
    save_filtered_annotations(train_split, data_dir / "train" / "annotations.json")
    save_filtered_annotations(test_split, data_dir / "test" / "annotations.json")

    logger.info("Done. Train/test split complete.")


if __name__ == "__main__":
    main()
