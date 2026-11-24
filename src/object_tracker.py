#!/usr/bin/env python3
"""
AURANAV - Object Tracker
Multi-object tracking using SORT-like algorithm
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class TrackState:
    """Track state enumeration"""
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class Track:
    """Single object track"""

    def __init__(self, detection, track_id, max_age=30, min_hits=3):
        """
        Initialize track

        Args:
            detection: Initial detection (bbox, class_id, confidence)
            track_id: Unique track ID
            max_age: Maximum frames to keep without update
            min_hits: Minimum detections before confirmed
        """
        self.track_id = track_id
        self.class_id = detection['class_id']
        self.class_name = detection['class_name']
        self.max_age = max_age
        self.min_hits = min_hits

        # Initialize Kalman filter for bbox tracking
        self.kf = self._init_kalman_filter(detection['bbox'])

        # State
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        # History
        self.bbox_history = [detection['bbox']]
        self.confidence_history = [detection['confidence']]

    def _init_kalman_filter(self, bbox):
        """Initialize Kalman filter for bbox tracking"""
        # State: [x, y, w, h, dx, dy, dw, dh]
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])

        # Measurement noise
        kf.R *= 1.0

        # Process noise
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        # Initial state
        x, y, w, h = bbox
        kf.x[:4] = np.array([x, y, w, h]).reshape(4, 1)
        return kf

    def predict(self):
        """Predict next state"""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection):
        """
        Update track with detection

        Args:
            detection: Detection dictionary
        """
        bbox = detection['bbox']
        self.kf.update(bbox)

        self.hits += 1
        self.time_since_update = 0

        # Update state
        if self.state == TrackState.TENTATIVE and self.hits >= self.min_hits:
            self.state = TrackState.CONFIRMED

        # Store history
        self.bbox_history.append(bbox)
        self.confidence_history.append(detection['confidence'])

        # Keep only recent history
        if len(self.bbox_history) > 30:
            self.bbox_history = self.bbox_history[-30:]
            self.confidence_history = self.confidence_history[-30:]

    def get_bbox(self):
        """Get current bbox from Kalman filter"""
        return self.kf.x[:4].flatten()

    def get_velocity(self):
        """Get current velocity from Kalman filter"""
        return self.kf.x[4:].flatten()

    def is_confirmed(self):
        """Check if track is confirmed"""
        return self.state == TrackState.CONFIRMED

    def is_deleted(self):
        """Check if track should be deleted"""
        return self.state == TrackState.DELETED

    def mark_deleted(self):
        """Mark track as deleted"""
        self.state = TrackState.DELETED

    def get_size_change_rate(self):
        """Get rate of bbox size change (proxy for approaching/receding)"""
        if len(self.bbox_history) < 2:
            return 0.0

        # Compare recent size to older size
        recent_size = self.bbox_history[-1][2] * self.bbox_history[-1][3]
        old_size = self.bbox_history[0][2] * self.bbox_history[0][3]

        if old_size == 0:
            return 0.0

        # Normalized change per frame
        frames = len(self.bbox_history)
        change_rate = (recent_size - old_size) / (old_size * frames)

        return change_rate


class ObjectTracker:
    """Multi-object tracker"""

    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize tracker

        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before confirmed
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks = []
        self.next_id = 0

    def update(self, detections):
        """
        Update tracker with new detections

        Args:
            detections: List of detection dictionaries

        Returns:
            List of confirmed tracks
        """
        # Predict all tracks
        for track in self.tracks:
            track.predict()

        # Match detections to tracks
        matched_indices, unmatched_detections, unmatched_tracks = \
            self._match(detections)

        # Update matched tracks
        for det_idx, track_idx in matched_indices:
            self.tracks[track_idx].update(detections[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(detections[det_idx])

        # Mark old tracks for deletion
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            if track.time_since_update > self.max_age:
                track.mark_deleted()

        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Return confirmed tracks
        return [t for t in self.tracks if t.is_confirmed()]

    def _create_track(self, detection):
        """Create new track"""
        track = Track(
            detection,
            self.next_id,
            max_age=self.max_age,
            min_hits=self.min_hits
        )
        self.tracks.append(track)
        self.next_id += 1

    def _match(self, detections):
        """
        Match detections to tracks using Hungarian algorithm

        Args:
            detections: List of detections

        Returns:
            matched_indices: List of (det_idx, track_idx) pairs
            unmatched_detections: List of detection indices
            unmatched_tracks: List of track indices
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))

        # Compute IOU cost matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))

        for d, det in enumerate(detections):
            for t, track in enumerate(self.tracks):
                iou_matrix[d, t] = self._iou(det['bbox'], track.get_bbox())

        # Convert IOU to cost (higher IOU = lower cost)
        cost_matrix = 1.0 - iou_matrix

        # Hungarian algorithm
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        # Filter matches by IOU threshold
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        for d, t in zip(det_indices, track_indices):
            if iou_matrix[d, t] >= self.iou_threshold:
                matched_indices.append((d, t))
                unmatched_detections.remove(d)
                unmatched_tracks.remove(t)

        return matched_indices, unmatched_detections, unmatched_tracks

    def _iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union

        Args:
            bbox1: [x, y, w, h] in normalized coordinates
            bbox2: [x, y, w, h] in normalized coordinates

        Returns:
            IOU score
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to corners
        x1_min, y1_min = x1 - w1/2, y1 - h1/2
        x1_max, y1_max = x1 + w1/2, y1 + h1/2
        x2_min, y2_min = x2 - w2/2, y2 - h2/2
        x2_max, y2_max = x2 + w2/2, y2 + h2/2

        # Intersection
        x_inter_min = max(x1_min, x2_min)
        y_inter_min = max(y1_min, y2_min)
        x_inter_max = min(x1_max, x2_max)
        y_inter_max = min(y1_max, y2_max)

        if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
            return 0.0

        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
