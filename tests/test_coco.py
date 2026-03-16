"""Tests for COCO data utilities."""

from vision_demo.data.coco import (
    COCO_ID_TO_CONTIGUOUS,
    CONTIGUOUS_TO_LABEL,
    VEHICLE_CATEGORIES,
    filter_vehicle_annotations,
    get_image_download_list,
    split_dataset,
)


def _make_coco_json(annotations, images, categories=None):
    """Build a minimal COCO-format dict for testing."""
    if categories is None:
        categories = [{"id": cid, "name": name} for cid, name in VEHICLE_CATEGORIES.items()]
    return {"annotations": annotations, "images": images, "categories": categories}


class TestConstants:
    """Tests for COCO category constants."""

    def test_vehicle_category_count(self):
        """Eight vehicle categories are defined."""
        assert len(VEHICLE_CATEGORIES) == 8

    def test_contiguous_ids_are_zero_indexed(self):
        """Contiguous IDs run from 0 to 7."""
        assert set(COCO_ID_TO_CONTIGUOUS.values()) == set(range(8))

    def test_contiguous_labels_match_categories(self):
        """Every contiguous ID maps back to the correct label."""
        for coco_id, label in VEHICLE_CATEGORIES.items():
            contiguous_id = COCO_ID_TO_CONTIGUOUS[coco_id]
            assert CONTIGUOUS_TO_LABEL[contiguous_id] == label


class TestFilterVehicleAnnotations:
    """Tests for filter_vehicle_annotations."""

    def test_keeps_vehicle_annotations(self):
        """Vehicle annotations are retained."""
        anns = [{"id": 1, "image_id": 100, "category_id": 3, "bbox": [0, 0, 10, 10]}]
        images = [{"id": 100, "file_name": "img.jpg", "coco_url": "http://example.com/img.jpg"}]
        result = filter_vehicle_annotations(_make_coco_json(anns, images))
        assert len(result["annotations"]) == 1

    def test_drops_non_vehicle_annotations(self):
        """Non-vehicle annotations are excluded."""
        anns = [
            {"id": 1, "image_id": 100, "category_id": 3, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 100, "category_id": 1, "bbox": [0, 0, 5, 5]},  # person
            {"id": 3, "image_id": 101, "category_id": 18, "bbox": [0, 0, 5, 5]},  # dog
        ]
        images = [
            {"id": 100, "file_name": "a.jpg", "coco_url": "http://example.com/a.jpg"},
            {"id": 101, "file_name": "b.jpg", "coco_url": "http://example.com/b.jpg"},
        ]
        result = filter_vehicle_annotations(_make_coco_json(anns, images))
        assert len(result["annotations"]) == 1
        assert result["annotations"][0]["id"] == 1

    def test_drops_images_without_vehicles(self):
        """Images with no vehicle annotations are excluded."""
        anns = [
            {"id": 1, "image_id": 100, "category_id": 3, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 101, "category_id": 1, "bbox": [0, 0, 5, 5]},  # person only
        ]
        images = [
            {"id": 100, "file_name": "a.jpg", "coco_url": "http://example.com/a.jpg"},
            {"id": 101, "file_name": "b.jpg", "coco_url": "http://example.com/b.jpg"},
        ]
        result = filter_vehicle_annotations(_make_coco_json(anns, images))
        assert len(result["images"]) == 1
        assert result["images"][0]["id"] == 100

    def test_remaps_category_ids(self):
        """Category IDs are remapped to contiguous 0-7."""
        anns = [
            {"id": 1, "image_id": 100, "category_id": 2, "bbox": [0, 0, 10, 10]},  # bicycle → 0
            {"id": 2, "image_id": 100, "category_id": 9, "bbox": [0, 0, 5, 5]},  # boat → 7
        ]
        images = [{"id": 100, "file_name": "a.jpg", "coco_url": "http://example.com/a.jpg"}]
        result = filter_vehicle_annotations(_make_coco_json(anns, images))
        ids = {a["category_id"] for a in result["annotations"]}
        assert ids == {0, 7}

    def test_does_not_mutate_input(self):
        """Original annotations are not modified."""
        anns = [{"id": 1, "image_id": 100, "category_id": 3, "bbox": [0, 0, 10, 10]}]
        images = [{"id": 100, "file_name": "a.jpg", "coco_url": "http://example.com/a.jpg"}]
        coco_json = _make_coco_json(anns, images)
        filter_vehicle_annotations(coco_json)
        assert coco_json["annotations"][0]["category_id"] == 3

    def test_empty_annotations(self):
        """Empty input produces empty output."""
        result = filter_vehicle_annotations(_make_coco_json([], []))
        assert result["annotations"] == []
        assert result["images"] == []

    def test_categories_are_remapped(self):
        """Output categories use contiguous IDs and correct names."""
        result = filter_vehicle_annotations(_make_coco_json([], []))
        cats = {c["id"]: c["name"] for c in result["categories"]}
        assert cats == CONTIGUOUS_TO_LABEL


class TestGetImageDownloadList:
    """Tests for get_image_download_list."""

    def test_extracts_required_fields(self):
        """Download list contains id, file_name, and coco_url."""
        filtered = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "coco_url": "http://example.com/a.jpg", "height": 480, "width": 640},
            ],
            "annotations": [],
            "categories": [],
        }
        result = get_image_download_list(filtered)
        assert len(result) == 1
        assert set(result[0].keys()) == {"id", "file_name", "coco_url"}

    def test_empty_images(self):
        """Empty image list returns empty download list."""
        result = get_image_download_list({"images": [], "annotations": [], "categories": []})
        assert result == []


def _make_filtered_coco(n_images: int) -> dict:
    """Build a filtered COCO dict with n images, each having one annotation."""
    images = [{"id": i, "file_name": f"{i}.jpg", "coco_url": f"http://example.com/{i}.jpg"} for i in range(n_images)]
    anns = [{"id": i, "image_id": i, "category_id": 0, "bbox": [0, 0, 10, 10]} for i in range(n_images)]
    categories = [{"id": 0, "name": "bicycle"}]
    return {"images": images, "annotations": anns, "categories": categories}


class TestSplitDataset:
    """Tests for split_dataset."""

    def test_split_sizes(self):
        """Split produces correct train/test sizes."""
        coco = _make_filtered_coco(100)
        train, test = split_dataset(coco, test_ratio=0.15, seed=42)
        assert len(test["images"]) == 15
        assert len(train["images"]) == 85

    def test_no_image_overlap(self):
        """Train and test sets have no overlapping image IDs."""
        coco = _make_filtered_coco(100)
        train, test = split_dataset(coco, test_ratio=0.15, seed=42)
        train_ids = {img["id"] for img in train["images"]}
        test_ids = {img["id"] for img in test["images"]}
        assert train_ids.isdisjoint(test_ids)

    def test_all_images_preserved(self):
        """All original images appear in exactly one split."""
        coco = _make_filtered_coco(100)
        train, test = split_dataset(coco, test_ratio=0.15, seed=42)
        train_ids = {img["id"] for img in train["images"]}
        test_ids = {img["id"] for img in test["images"]}
        original_ids = {img["id"] for img in coco["images"]}
        assert train_ids | test_ids == original_ids

    def test_annotations_follow_images(self):
        """Annotations are assigned to the same split as their image."""
        coco = _make_filtered_coco(100)
        train, test = split_dataset(coco, test_ratio=0.15, seed=42)
        test_image_ids = {img["id"] for img in test["images"]}
        for ann in test["annotations"]:
            assert ann["image_id"] in test_image_ids
        train_image_ids = {img["id"] for img in train["images"]}
        for ann in train["annotations"]:
            assert ann["image_id"] in train_image_ids

    def test_deterministic_with_same_seed(self):
        """Same seed produces the same split."""
        coco = _make_filtered_coco(100)
        train1, test1 = split_dataset(coco, test_ratio=0.15, seed=42)
        train2, test2 = split_dataset(coco, test_ratio=0.15, seed=42)
        assert [img["id"] for img in test1["images"]] == [img["id"] for img in test2["images"]]

    def test_different_seed_produces_different_split(self):
        """Different seeds produce different splits."""
        coco = _make_filtered_coco(100)
        _, test1 = split_dataset(coco, test_ratio=0.15, seed=42)
        _, test2 = split_dataset(coco, test_ratio=0.15, seed=99)
        assert [img["id"] for img in test1["images"]] != [img["id"] for img in test2["images"]]

    def test_categories_preserved(self):
        """Both splits retain the same categories."""
        coco = _make_filtered_coco(100)
        train, test = split_dataset(coco, test_ratio=0.15, seed=42)
        assert train["categories"] == coco["categories"]
        assert test["categories"] == coco["categories"]

    def test_does_not_mutate_input(self):
        """Original data is not modified."""
        coco = _make_filtered_coco(100)
        original_count = len(coco["images"])
        split_dataset(coco, test_ratio=0.15, seed=42)
        assert len(coco["images"]) == original_count
