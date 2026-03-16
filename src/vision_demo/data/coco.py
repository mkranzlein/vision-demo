"""COCO dataset utilities for vehicle detection.

Provides functions to download COCO 2017 annotations, filter to vehicle
classes, and download the corresponding images.
"""

import asyncio
import json
import logging
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import aiohttp

logger = logging.getLogger(__name__)

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

VEHICLE_CATEGORIES = {
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
}

COCO_ID_TO_CONTIGUOUS = {coco_id: idx for idx, coco_id in enumerate(sorted(VEHICLE_CATEGORIES))}
"""Maps sparse COCO category IDs (2-9) to contiguous IDs (0-7)."""

CONTIGUOUS_TO_LABEL = {idx: VEHICLE_CATEGORIES[coco_id] for coco_id, idx in COCO_ID_TO_CONTIGUOUS.items()}
"""Maps contiguous IDs (0-7) to human-readable labels."""


def filter_vehicle_annotations(coco_json: dict) -> dict:
    """Filter a COCO annotation dict to only vehicle classes.

    Args:
        coco_json: Parsed COCO annotation JSON with 'annotations', 'images',
            and 'categories' keys.

    Returns:
        A new dict in COCO format containing only vehicle annotations,
        their associated images, and vehicle categories with remapped IDs.
    """
    vehicle_cat_ids = set(VEHICLE_CATEGORIES.keys())

    vehicle_anns = [a for a in coco_json["annotations"] if a["category_id"] in vehicle_cat_ids]

    image_ids_with_vehicles = {a["image_id"] for a in vehicle_anns}
    vehicle_images = [img for img in coco_json["images"] if img["id"] in image_ids_with_vehicles]

    remapped_anns = []
    for ann in vehicle_anns:
        remapped = dict(ann)
        remapped["category_id"] = COCO_ID_TO_CONTIGUOUS[ann["category_id"]]
        remapped_anns.append(remapped)

    vehicle_cats = [
        {"id": COCO_ID_TO_CONTIGUOUS[cid], "name": name} for cid, name in sorted(VEHICLE_CATEGORIES.items())
    ]

    return {
        "images": vehicle_images,
        "annotations": remapped_anns,
        "categories": vehicle_cats,
    }


def get_image_download_list(filtered_coco: dict) -> list[dict]:
    """Extract image URLs and filenames from filtered COCO metadata.

    Args:
        filtered_coco: COCO-format dict (output of filter_vehicle_annotations).

    Returns:
        List of dicts with 'id', 'file_name', and 'coco_url' keys.
    """
    return [
        {"id": img["id"], "file_name": img["file_name"], "coco_url": img["coco_url"]} for img in filtered_coco["images"]
    ]


def download_annotations(dest_dir: Path) -> tuple[Path, Path]:
    """Download and extract COCO 2017 train/val annotation files.

    Args:
        dest_dir: Directory to save annotation files.

    Returns:
        Tuple of (train_ann_path, val_ann_path).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "annotations_trainval2017.zip"

    if not zip_path.exists():
        logger.info("Downloading COCO 2017 annotations (~252 MB)...")
        urllib.request.urlretrieve(ANNOTATIONS_URL, zip_path)  # noqa: S310
        logger.info("Download complete.")
    else:
        logger.info("Annotations zip already exists, skipping download.")

    train_path = dest_dir / "instances_train2017.json"
    val_path = dest_dir / "instances_val2017.json"

    if not train_path.exists() or not val_path.exists():
        logger.info("Extracting annotations...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith("instances_train2017.json") or member.endswith("instances_val2017.json"):
                    data = zf.read(member)
                    out_name = Path(member).name
                    (dest_dir / out_name).write_bytes(data)
        logger.info("Extraction complete.")

    return train_path, val_path


async def _download_single_image(
    session: aiohttp.ClientSession, img: dict, dest_dir: Path, sem: asyncio.Semaphore
) -> bool:
    """Download a single image if it doesn't already exist.

    Args:
        session: aiohttp client session.
        img: Dict with 'file_name' and 'coco_url' keys.
        dest_dir: Directory to save the image.
        sem: Semaphore to limit concurrency.

    Returns:
        True if the image was downloaded, False if it already existed.
    """
    dest_path = dest_dir / img["file_name"]
    if dest_path.exists():
        return False
    async with sem, session.get(img["coco_url"]) as resp:
        data = await resp.read()
    dest_path.write_bytes(data)
    return True


async def _download_images_async(image_list: list[dict], dest_dir: Path, max_concurrent: int = 32) -> int:
    """Download COCO images concurrently with aiohttp.

    Args:
        image_list: List of dicts with 'file_name' and 'coco_url' keys.
        dest_dir: Directory to save images.
        max_concurrent: Maximum number of concurrent downloads.

    Returns:
        Number of images downloaded (excluding skipped).
    """
    sem = asyncio.Semaphore(max_concurrent)
    async with aiohttp.ClientSession() as session:
        tasks = [_download_single_image(session, img, dest_dir, sem) for img in image_list]
        results = await asyncio.gather(*tasks)

    downloaded = sum(results)
    skipped = len(results) - downloaded
    logger.info("Downloaded %d new images (%d already existed).", downloaded, skipped)
    return downloaded


def download_images(image_list: list[dict], dest_dir: Path, max_concurrent: int = 32) -> int:
    """Download COCO images to a local directory using async I/O.

    Skips images that already exist on disk.

    Args:
        image_list: List of dicts with 'file_name' and 'coco_url' keys
            (output of get_image_download_list).
        dest_dir: Directory to save images.
        max_concurrent: Maximum number of concurrent downloads.

    Returns:
        Number of images downloaded (excluding skipped).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_download_images_async(image_list, dest_dir, max_concurrent))


def split_dataset(filtered_coco: dict, test_ratio: float = 0.15, seed: int = 42) -> tuple[dict, dict]:
    """Split a COCO-format dataset into train and test sets by image ID.

    Args:
        filtered_coco: COCO-format dict with 'images', 'annotations', and 'categories'.
        test_ratio: Fraction of images to hold out for test.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_coco, test_coco) dicts in COCO format.
    """
    images = list(filtered_coco["images"])
    rng = random.Random(seed)
    rng.shuffle(images)

    split_idx = int(len(images) * test_ratio)
    test_images = images[:split_idx]
    train_images = images[split_idx:]

    test_image_ids = {img["id"] for img in test_images}

    train_anns = [a for a in filtered_coco["annotations"] if a["image_id"] not in test_image_ids]
    test_anns = [a for a in filtered_coco["annotations"] if a["image_id"] in test_image_ids]

    categories = filtered_coco["categories"]
    return (
        {"images": train_images, "annotations": train_anns, "categories": categories},
        {"images": test_images, "annotations": test_anns, "categories": categories},
    )


def split_image_files(train_coco: dict, test_coco: dict, source_images_dir: Path, test_images_dir: Path) -> int:
    """Move test images from the train directory to a separate test directory.

    Args:
        train_coco: Train split COCO-format dict (unused, for API symmetry).
        test_coco: Test split COCO-format dict.
        source_images_dir: Directory currently containing all images.
        test_images_dir: Destination directory for test images.

    Returns:
        Number of images moved.
    """
    test_images_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for img in test_coco["images"]:
        src = source_images_dir / img["file_name"]
        dst = test_images_dir / img["file_name"]
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))
            moved += 1
    logger.info("Moved %d images to %s.", moved, test_images_dir)
    return moved


def save_filtered_annotations(filtered_coco: dict, dest_path: Path) -> None:
    """Write filtered COCO annotations to a JSON file.

    Args:
        filtered_coco: COCO-format dict to save.
        dest_path: Output file path.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(json.dumps(filtered_coco))
    logger.info(
        "Saved %d annotations for %d images to %s.",
        len(filtered_coco["annotations"]),
        len(filtered_coco["images"]),
        dest_path,
    )
