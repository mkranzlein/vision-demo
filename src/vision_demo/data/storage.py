"""MinIO storage client for uploading and downloading dataset files."""

import logging
from pathlib import Path

from minio import Minio

logger = logging.getLogger(__name__)

DEFAULT_ENDPOINT = "localhost:9000"
DEFAULT_BUCKET = "vision-demo"
DEFAULT_ACCESS_KEY = "minioadmin"
DEFAULT_SECRET_KEY = "minioadmin"


def get_client(
    endpoint: str = DEFAULT_ENDPOINT,
    access_key: str = DEFAULT_ACCESS_KEY,
    secret_key: str = DEFAULT_SECRET_KEY,
) -> Minio:
    """Create a MinIO client.

    Args:
        endpoint: MinIO server address.
        access_key: Access key.
        secret_key: Secret key.

    Returns:
        A configured Minio client.
    """
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=False)


def upload_directory(
    client: Minio,
    local_dir: Path,
    bucket: str,
    prefix: str,
) -> int:
    """Upload all files in a local directory to MinIO.

    Args:
        client: MinIO client.
        local_dir: Local directory to upload.
        bucket: Target bucket name.
        prefix: Object name prefix (e.g. "train/images").

    Returns:
        Number of files uploaded.
    """
    uploaded = 0
    for file_path in sorted(local_dir.rglob("*")):
        if not file_path.is_file():
            continue
        object_name = f"{prefix}/{file_path.relative_to(local_dir)}"
        client.fput_object(bucket, object_name, str(file_path))
        uploaded += 1
        if uploaded % 500 == 0:
            logger.info("Uploaded %d files...", uploaded)
    logger.info("Uploaded %d files to %s/%s.", uploaded, bucket, prefix)
    return uploaded


def download_directory(
    client: Minio,
    bucket: str,
    prefix: str,
    local_dir: Path,
) -> int:
    """Download all objects under a prefix from MinIO to a local directory.

    Skips files that already exist locally.

    Args:
        client: MinIO client.
        bucket: Source bucket name.
        prefix: Object name prefix to download.
        local_dir: Local directory to save files.

    Returns:
        Number of files downloaded.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    for obj in client.list_objects(bucket, prefix=prefix, recursive=True):
        rel_path = obj.object_name.removeprefix(prefix).lstrip("/")
        local_path = local_dir / rel_path
        if local_path.exists():
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(bucket, obj.object_name, str(local_path))
        downloaded += 1
        if downloaded % 500 == 0:
            logger.info("Downloaded %d files...", downloaded)
    logger.info("Downloaded %d files from %s/%s.", downloaded, bucket, prefix)
    return downloaded
