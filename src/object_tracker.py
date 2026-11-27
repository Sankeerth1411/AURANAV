#!/usr/bin/env python3
"""
AURANAV class-aware multi-object tracker.

Boxes use normalized center format [cx, cy, width, height].
This lightweight IoU tracker avoids Kalman prediction drift and keeps
objects of different classes from being associated with one another.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    bbox: np.ndarray
    confidence: float
    min_hits: int = 2
    max_age: int = 8
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confirmed: bool = False
    bbox_history: List[np.ndarray] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=float).reshape(4)
        self.bbox_history.append(self.bbox.copy())
        self.confidence_history.append(float(self.confidence))
        self.confirmed = self.hits >= self.min_hits

    def predict(self):
        # Do not extrapolate boxes without a motion model; mark them stale instead.
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Dict):
        self.bbox = np.asarray(detection["bbox"], dtype=float).reshape(4)
        self.class_id = int(detection["class_id"])
        self.class_name = str(detection["class_name"])
        self.confidence = float(detection["confidence"])
        self.hits += 1
        self.time_since_update = 0
        self.confirmed = self.hits >= self.min_hits
        self.bbox_history.append(self.bbox.copy())
        self.confidence_history.append(self.confidence)
        self.bbox_history = self.bbox_history[-12:]
        self.confidence_history = self.confidence_history[-12:]

    def get_bbox(self):
        return self.bbox.copy()

    def get_velocity(self):
        if len(self.bbox_history) < 2:
            return np.zeros(4, dtype=float)
        return self.bbox_history[-1] - self.bbox_history[-2]

    def get_size_change_rate(self):
        if len(self.bbox_history) < 2:
            return 0.0
        areas = [max(0.0, float(b[2] * b[3])) for b in self.bbox_history[-6:]]
        if len(areas) < 2 or areas[0] <= 1e-9:
            return 0.0
        return (areas[-1] - areas[0]) / (areas[0] * (len(areas) - 1))

    def is_confirmed(self):
        return self.confirmed

    def is_deleted(self):
        return self.time_since_update > self.max_age

    def mark_deleted(self):
        self.time_since_update = self.max_age + 1


class ObjectTracker:
    """Class-aware IoU tracker with bounded stale-track lifetime."""

    def __init__(self, max_age=8, min_hits=2, iou_threshold=0.3):
        self.max_age = max(1, int(max_age))
        self.min_hits = max(1, int(min_hits))
        self.iou_threshold = float(iou_threshold)
        self.tracks: List[Track] = []
        self.next_id = 0

    @staticmethod
    def _valid_detection(detection):
        try:
            bbox = np.asarray(detection["bbox"], dtype=float).reshape(4)
            confidence = float(detection["confidence"])
            int(detection["class_id"])
            return (
                np.all(np.isfinite(bbox))
                and bbox[2] > 0 and bbox[3] > 0
                and np.all(bbox[:2] >= 0) and np.all(bbox[:2] <= 1)
                and np.all(bbox[2:] <= 1)
                and np.isfinite(confidence) and 0 <= confidence <= 1
            )
        except (KeyError, TypeError, ValueError):
            return False

    def update(self, detections):
        detections = [d for d in detections if self._valid_detection(d)]
        for track in self.tracks:
            track.predict()

        matches, unmatched_dets, unmatched_tracks = self._match(detections)
        for det_idx, track_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        for det_idx in unmatched_dets:
            d = detections[det_idx]
            self.tracks.append(Track(
                track_id=self.next_id,
                class_id=int(d["class_id"]),
                class_name=str(d["class_name"]),
                bbox=d["bbox"],
                confidence=float(d["confidence"]),
                min_hits=self.min_hits,
                max_age=self.max_age,
            ))
            self.next_id += 1

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        # Only confirmed, currently observed tracks are exposed to the hazard engine.
        return [t for t in self.tracks if t.is_confirmed() and t.time_since_update == 0]

    def _match(self, detections):
        if not self.tracks:
            return [], list(range(len(detections))), []
        if not detections:
            return [], [], list(range(len(self.tracks)))

        iou = np.zeros((len(detections), len(self.tracks)), dtype=float)
        for di, detection in enumerate(detections):
            for ti, track in enumerate(self.tracks):
                if int(detection["class_id"]) != track.class_id:
                    continue
                iou[di, ti] = self._iou(detection["bbox"], track.get_bbox())

        rows, cols = linear_sum_assignment(1.0 - iou)
        matches = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))
        for di, ti in zip(rows, cols):
            if iou[di, ti] >= self.iou_threshold:
                matches.append((int(di), int(ti)))
                unmatched_dets.discard(int(di))
                unmatched_tracks.discard(int(ti))
        return matches, sorted(unmatched_dets), sorted(unmatched_tracks)

    @staticmethod
    def _iou(bbox1, bbox2):
        a = np.asarray(bbox1, dtype=float).reshape(4)
        b = np.asarray(bbox2, dtype=float).reshape(4)
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return 0.0
        ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
        ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
        bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
        bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = iw * ih
        union = max(0.0, a[2] * a[3]) + max(0.0, b[2] * b[3]) - intersection
        return intersection / union if union > 1e-12 else 0.0
