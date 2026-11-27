"""Regression tests for AURANAV's safety-oriented tracking and scoring."""

import numpy as np

from src.object_tracker import ObjectTracker
from src.hazard_scorer import HazardScorer


def detection(class_id=1, name="person", bbox=(0.5, 0.6, 0.2, 0.3), confidence=0.9):
    return {
        "class_id": class_id,
        "class_name": name,
        "bbox": list(bbox),
        "confidence": confidence,
    }


def test_tracker_confirms_only_after_required_hits():
    tracker = ObjectTracker(min_hits=2, max_age=2, iou_threshold=0.2)
    assert tracker.update([detection()]) == []
    tracks = tracker.update([detection(bbox=(0.51, 0.6, 0.2, 0.3))])
    assert len(tracks) == 1
    assert tracks[0].is_confirmed()


def test_tracker_never_matches_different_classes():
    tracker = ObjectTracker(min_hits=1, iou_threshold=0.1)
    first = tracker.update([detection(class_id=1, name="person")])
    first_id = first[0].track_id
    second = tracker.update([detection(class_id=2, name="car")])
    assert len(second) == 1
    assert second[0].class_name == "car"
    assert second[0].track_id != first_id


def test_tracker_does_not_return_stale_tracks_as_current():
    tracker = ObjectTracker(min_hits=1, max_age=1)
    assert len(tracker.update([detection()])) == 1
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    assert tracker.tracks == []


def test_tracker_rejects_invalid_boxes():
    tracker = ObjectTracker(min_hits=1)
    tracks = tracker.update([detection(bbox=(float("nan"), 0.5, 0.2, 0.2))])
    assert tracks == []


class FakeTrack:
    track_id = 7
    class_id = 0
    class_name = "pothole"
    confidence_history = [0.9, 0.95]

    def __init__(self, bbox):
        self.bbox = np.asarray(bbox, dtype=float)

    def get_bbox(self):
        return self.bbox

    def get_size_change_rate(self):
        return 0.0


def test_hazard_scorer_handles_invalid_bbox_safely():
    scorer = HazardScorer()
    track = FakeTrack([0.5, 0.5, -0.1, 0.2])
    assert scorer.score_track(track, 640, 480) == 0.0


def test_hazard_scorer_returns_bounded_score_and_direction():
    scorer = HazardScorer()
    track = FakeTrack([0.5, 0.75, 0.4, 0.4])
    score = scorer.score_track(track, 640, 480)
    assert 0.0 <= score <= 100.0
    assert scorer.get_direction(track.get_bbox()) == "ahead"
