#!/usr/bin/env python3
"""
AURANAV - Simple test suite
Basic tests for core functionality
"""

import sys
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


class TestConfigLoader:
    """Test configuration loading"""

    def test_config_loads(self):
        """Test that config loads without errors"""
        from src.config_loader import get_config
        config = get_config()
        assert config is not None

    def test_config_sections_exist(self):
        """Test required config sections exist"""
        from src.config_loader import get_config
        config = get_config()

        required = ['roboflow', 'paths', 'model', 'training',
                   'inference', 'camera', 'hazard', 'audio']

        for section in required:
            assert section in config.config


class TestHazardScorer:
    """Test hazard scoring logic"""

    def test_scorer_initialization(self):
        """Test hazard scorer initializes"""
        from src.hazard_scorer import HazardScorer
        scorer = HazardScorer()
        assert scorer is not None

    def test_class_severity(self):
        """Test class severity lookup"""
        from src.hazard_scorer import HazardScorer
        scorer = HazardScorer()

        # Test known hazardous class
        severity = scorer._get_class_severity('pothole')
        assert severity > 80

        # Test less hazardous class
        severity = scorer._get_class_severity('bird')
        assert severity < 50

    def test_position_multiplier(self):
        """Test position-based multiplier"""
        from src.hazard_scorer import HazardScorer
        scorer = HazardScorer()

        # Center position should have higher multiplier
        center_bbox = [0.5, 0.5, 0.1, 0.1]
        center_mult = scorer._get_position_multiplier(center_bbox, 640, 480)

        # Side position should have lower multiplier
        side_bbox = [0.1, 0.5, 0.1, 0.1]
        side_mult = scorer._get_position_multiplier(side_bbox, 640, 480)

        assert center_mult > side_mult

    def test_size_multiplier(self):
        """Test size-based multiplier"""
        from src.hazard_scorer import HazardScorer
        scorer = HazardScorer()

        # Large bbox should have higher multiplier
        large_bbox = [0.5, 0.5, 0.4, 0.4]
        large_mult = scorer._get_size_multiplier(large_bbox)

        # Small bbox should have lower multiplier
        small_bbox = [0.5, 0.5, 0.05, 0.05]
        small_mult = scorer._get_size_multiplier(small_bbox)

        assert large_mult > small_mult

    def test_direction_detection(self):
        """Test direction detection"""
        from src.hazard_scorer import HazardScorer
        scorer = HazardScorer()

        # Left
        left_bbox = [0.2, 0.5, 0.1, 0.1]
        assert scorer.get_direction(left_bbox) == 'left'

        # Center
        center_bbox = [0.5, 0.5, 0.1, 0.1]
        assert scorer.get_direction(center_bbox) == 'ahead'

        # Right
        right_bbox = [0.8, 0.5, 0.1, 0.1]
        assert scorer.get_direction(right_bbox) == 'right'


class TestObjectTracker:
    """Test object tracking"""

    def test_tracker_initialization(self):
        """Test tracker initializes"""
        from src.object_tracker import ObjectTracker
        tracker = ObjectTracker()
        assert tracker is not None
        assert len(tracker.tracks) == 0

    def test_iou_calculation(self):
        """Test IOU calculation"""
        from src.object_tracker import ObjectTracker
        tracker = ObjectTracker()

        # Same bbox should have IOU = 1.0
        bbox = [0.5, 0.5, 0.2, 0.2]
        iou = tracker._iou(bbox, bbox)
        assert abs(iou - 1.0) < 0.001

        # Non-overlapping bboxes should have IOU = 0.0
        bbox1 = [0.2, 0.2, 0.1, 0.1]
        bbox2 = [0.8, 0.8, 0.1, 0.1]
        iou = tracker._iou(bbox1, bbox2)
        assert iou == 0.0


def run_tests():
    """Run all tests"""
    pytest.main([__file__, '-v'])


if __name__ == "__main__":
    run_tests()
