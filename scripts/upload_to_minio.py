"""Upload local COCO vehicle data splits to MinIO.

Usage:
    python scripts/upload_to_minio.py
    python scripts/upload_to_minio.py --data-dir data/coco --endpoint localhost:9000
"""

import argparse
import logging
from pathlib import Path

from vision_demo.data.storage import DEFAULT_BUCKET, DEFAULT_ENDPOINT, get_client, upload_directory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "coco"


def main():
    """Upload data splits to MinIO."""
    parser = argparse.ArgumentParser(description="Upload COCO vehicle data to MinIO.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Local COCO data directory.")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="MinIO endpoint.")
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET, help="MinIO bucket name.")
    args = parser.parse_args()

    client = get_client(endpoint=args.endpoint)

    for split in ["train", "val", "test"]:
        split_dir = args.data_dir / split
        if not split_dir.exists():
            logger.warning("Split directory %s not found, skipping.", split_dir)
            continue

        # Upload images
        images_dir = split_dir / "images"
        if images_dir.exists():
            logger.info("Uploading %s images...", split)
            upload_directory(client, images_dir, args.bucket, f"data/{split}/images")

        # Upload annotations
        ann_path = split_dir / "annotations.json"
        if ann_path.exists():
            object_name = f"data/{split}/annotations.json"
            client.fput_object(args.bucket, object_name, str(ann_path))
            logger.info("Uploaded %s annotations.", split)

    logger.info("Done. Data uploaded to %s/%s.", args.endpoint, args.bucket)


if __name__ == "__main__":
    main()
