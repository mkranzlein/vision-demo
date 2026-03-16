"""Tests for the Faster R-CNN detector factory."""

import torch

from vision_demo.model.detector import NUM_CLASSES, create_model


class TestCreateModel:
    """Tests for create_model."""

    def test_model_output_classes(self):
        """Model classification head has the correct number of classes."""
        model = create_model(pretrained=False)
        head = model.roi_heads.box_predictor
        assert head.cls_score.out_features == NUM_CLASSES

    def test_custom_num_classes(self):
        """Model respects custom num_classes argument."""
        model = create_model(num_classes=5, pretrained=False)
        head = model.roi_heads.box_predictor
        assert head.cls_score.out_features == 5

    def test_model_trains(self):
        """Model runs a forward pass in training mode and returns losses."""
        model = create_model(pretrained=False)
        model.train()

        images = [torch.rand(3, 100, 100)]
        targets = [{"boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]), "labels": torch.tensor([1])}]

        loss_dict = model(images, targets)
        assert isinstance(loss_dict, dict)
        assert all(v.item() >= 0 for v in loss_dict.values())
