"""Tests for the optional open-vocabulary discovery channel."""

import numpy as np

from src.open_vocabulary import box_iou_xywh, remove_primary_duplicates, OpenVocabularyDiscovery


def test_box_iou_identical_boxes_is_one():
    box = [0.5, 0.5, 0.2, 0.2]
    assert np.isclose(box_iou_xywh(box, box), 1.0, atol=1e-12)


def test_box_iou_non_overlapping_boxes_is_zero():
    assert box_iou_xywh([0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]) == 0.0


def test_box_iou_invalid_boxes_is_zero():
    assert box_iou_xywh([np.nan, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]) == 0.0
    assert box_iou_xywh([0.5, 0.5, -0.2, 0.2], [0.5, 0.5, 0.2, 0.2]) == 0.0


def test_primary_detector_overlap_suppresses_discovery_duplicate():
    primary = [{"bbox": [0.5, 0.5, 0.2, 0.2]}]
    candidates = [
        {"bbox": [0.5, 0.5, 0.2, 0.2], "class_name": "person"},
        {"bbox": [0.9, 0.9, 0.05, 0.05], "class_name": "open drain"},
    ]
    kept = remove_primary_duplicates(candidates, primary)
    assert len(kept) == 1
    assert kept[0]["class_name"] == "open drain"


def test_discovery_is_disabled_by_default_without_loading_model():
    discovery = OpenVocabularyDiscovery({"enabled": False})
    assert discovery.available is False
    assert discovery.predict(np.zeros((64, 64, 3), dtype=np.uint8)) == []
