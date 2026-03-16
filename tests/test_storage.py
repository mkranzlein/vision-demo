"""Tests for MinIO storage client."""

from unittest.mock import MagicMock

from vision_demo.data.storage import download_directory, get_client, upload_directory


class TestGetClient:
    """Tests for get_client."""

    def test_returns_minio_client(self):
        """Client is created with default settings."""
        client = get_client()
        assert "localhost" in client._base_url.host

    def test_custom_endpoint(self):
        """Client respects custom endpoint."""
        client = get_client(endpoint="minio:9000")
        assert "minio" in client._base_url.host


class TestUploadDirectory:
    """Tests for upload_directory."""

    def test_uploads_files(self, tmp_path):
        """All files in directory are uploaded."""
        (tmp_path / "a.jpg").write_text("img")
        (tmp_path / "b.jpg").write_text("img")

        client = MagicMock()
        count = upload_directory(client, tmp_path, "bucket", "prefix")
        assert count == 2
        assert client.fput_object.call_count == 2

    def test_skips_subdirectories(self, tmp_path):
        """Directories themselves are not uploaded."""
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "file.txt").write_text("data")

        client = MagicMock()
        count = upload_directory(client, tmp_path, "bucket", "prefix")
        assert count == 1

    def test_object_names_include_prefix(self, tmp_path):
        """Uploaded object names combine prefix and relative path."""
        (tmp_path / "img.jpg").write_text("img")

        client = MagicMock()
        upload_directory(client, tmp_path, "bucket", "train/images")
        client.fput_object.assert_called_once_with("bucket", "train/images/img.jpg", str(tmp_path / "img.jpg"))


class TestDownloadDirectory:
    """Tests for download_directory."""

    def test_downloads_files(self, tmp_path):
        """Objects under prefix are downloaded."""
        obj1 = MagicMock()
        obj1.object_name = "data/train/images/a.jpg"
        obj2 = MagicMock()
        obj2.object_name = "data/train/images/b.jpg"

        client = MagicMock()
        client.list_objects.return_value = [obj1, obj2]

        count = download_directory(client, "bucket", "data/train/images", tmp_path)
        assert count == 2
        assert client.fget_object.call_count == 2

    def test_skips_existing_files(self, tmp_path):
        """Files that already exist locally are skipped."""
        (tmp_path / "a.jpg").write_text("existing")

        obj = MagicMock()
        obj.object_name = "prefix/a.jpg"

        client = MagicMock()
        client.list_objects.return_value = [obj]

        count = download_directory(client, "bucket", "prefix", tmp_path)
        assert count == 0
        client.fget_object.assert_not_called()
