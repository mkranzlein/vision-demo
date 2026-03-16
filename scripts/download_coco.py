"""Download COCO 2017 vehicle images and annotations.

This script downloads the COCO 2017 annotation files, filters to vehicle
classes, and downloads the corresponding images. Data is saved to data/coco/.

Usage:
    python scripts/download_coco.py
    python scripts/download_coco.py --data-dir /path/to/data
"""

import argparse
import json
import logging
from pathlib import Path

from vision_demo.data.coco import (
    download_annotations,
    download_images,
    filter_vehicle_annotations,
    get_image_download_list,
    save_filtered_annotations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "coco"


def main():
    """Download COCO vehicle subset."""
    parser = argparse.ArgumentParser(description="Download COCO 2017 vehicle images and annotations.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Root directory for COCO data.")
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    ann_dir = data_dir / "annotations"

    # Step 1: Download annotation files
    logger.info("Step 1: Downloading COCO 2017 annotations...")
    train_ann_path, val_ann_path = download_annotations(ann_dir)

    # Step 2: Filter to vehicle classes
    logger.info("Step 2: Filtering annotations to vehicle classes...")
    for split, ann_path in [("train", train_ann_path), ("val", val_ann_path)]:
        logger.info("Processing %s split...", split)
        with ann_path.open() as f:
            coco_json = json.load(f)

        filtered = filter_vehicle_annotations(coco_json)
        logger.info("  %s: %d images, %d annotations", split, len(filtered["images"]), len(filtered["annotations"]))

        # Save filtered annotations
        save_filtered_annotations(filtered, data_dir / split / "annotations.json")

        # Step 3: Download images
        logger.info("Step 3: Downloading %s images...", split)
        image_list = get_image_download_list(filtered)
        download_images(image_list, data_dir / split / "images")

    logger.info("Done. Data saved to %s", data_dir)


if __name__ == "__main__":
    main()
