#!/usr/bin/env python3
"""
AURANAV - Hazard Scoring Engine
Intelligent risk assessment for detected objects
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import get_config


@dataclass
class HazardAlert:
    """Hazard alert information"""
    track_id: int
    class_name: str
    score: float
    level: str  # 'critical', 'warning', 'info'
    direction: str  # 'left', 'ahead', 'right'
    message: str
    timestamp: float


class HazardScorer:
    """Score hazard risk of detected objects"""

    def __init__(self):
        """Initialize hazard scorer"""
        self.config = get_config()
        self.hazard_config = self.config.hazard_config

        # Alert history for cooldown management
        self.last_alert_time = {}  # track_id -> timestamp
        self.last_class_alert = {}  # class_name -> timestamp
        self.last_any_alert = 0.0

    def score_track(self, track, frame_width: int, frame_height: int) -> float:
        """
        Compute hazard score for a tracked object

        Args:
            track: Track object
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Hazard score (0-100)
        """
        # Base severity from class
        base_score = self._get_class_severity(track.class_name)

        # Position multiplier
        bbox = track.get_bbox()
        position_mult = self._get_position_multiplier(bbox, frame_width, frame_height)

        # Size multiplier (proxy for proximity)
        size_mult = self._get_size_multiplier(bbox)

        # Movement multiplier
        movement_mult = self._get_movement_multiplier(track)

        # Confidence factor
        avg_confidence = np.mean(track.confidence_history[-5:])  # Last 5 detections
        confidence_mult = 0.5 + (avg_confidence * 0.5)  # Range: 0.5 - 1.0

        # Compute final score
        score = base_score * position_mult * size_mult * movement_mult * confidence_mult

        # Clamp to valid range
        score = np.clip(score, 0, 100)

        return float(score)

    def _get_class_severity(self, class_name: str) -> float:
        """Get base severity score for class"""
        severity_map = self.hazard_config['class_severity']

        # Direct match
        if class_name in severity_map:
            return float(severity_map[class_name])

        # Check for vehicle classes
        vehicle_classes = ['car', 'truck', 'bus']
        if class_name in vehicle_classes:
            return float(severity_map.get('vehicle', 80))

        # Default severity
        return 50.0

    def _get_position_multiplier(self, bbox, frame_width: int, frame_height: int) -> float:
        """
        Get position-based multiplier

        Center path is most critical for walking
        """
        x_center = bbox[0]  # Normalized x coordinate

        position_config = self.hazard_config['position']
        center_width = position_config['center_zone_width']
        center_mult = position_config['center_multiplier']
        side_mult = position_config['side_multiplier']

        # Check if in center zone
        center_left = 0.5 - center_width / 2
        center_right = 0.5 + center_width / 2

        if center_left <= x_center <= center_right:
            return center_mult
        else:
            return side_mult

    def _get_size_multiplier(self, bbox) -> float:
        """
        Get size-based multiplier (proxy for proximity)

        Larger bbox = closer = more dangerous
        Note: This is NOT exact distance, just relative proximity
        """
        # Compute bbox area (normalized)
        width, height = bbox[2], bbox[3]
        area = width * height

        size_config = self.hazard_config['size']

        if area >= size_config['large_threshold']:
            return size_config['xlarge_multiplier']
        elif area >= size_config['medium_threshold']:
            return size_config['large_multiplier']
        elif area >= size_config['small_threshold']:
            return size_config['medium_multiplier']
        else:
            return size_config['small_multiplier']

    def _get_movement_multiplier(self, track) -> float:
        """
        Get movement-based multiplier

        Approaching objects are more dangerous
        """
        size_change_rate = track.get_size_change_rate()

        movement_config = self.hazard_config['movement']
        velocity_threshold = movement_config['velocity_threshold']

        if size_change_rate > velocity_threshold:
            # Object getting larger = approaching
            return movement_config['approaching_multiplier']
        elif size_change_rate < -velocity_threshold:
            # Object getting smaller = receding
            return movement_config['receding_multiplier']
        else:
            # Stationary or slow movement
            return movement_config['stationary_multiplier']

    def get_direction(self, bbox) -> str:
        """
        Get direction label based on bbox position

        Args:
            bbox: Bounding box [x, y, w, h] in normalized coordinates

        Returns:
            Direction string: 'left', 'ahead', 'right'
        """
        x_center = bbox[0]

        direction_config = self.hazard_config['alert']['directions'] if 'directions' in self.hazard_config['alert'] else self.config.audio_config['directions']

        left_threshold = direction_config['left_threshold']
        right_threshold = direction_config['right_threshold']

        if x_center < left_threshold:
            return direction_config['left_label']
        elif x_center > right_threshold:
            return direction_config['right_label']
        else:
            return direction_config['center_label']

    def should_alert(self, track_id: int, class_name: str) -> bool:
        """
        Check if alert should be issued based on cooldown rules

        Args:
            track_id: Track ID
            class_name: Object class name

        Returns:
            True if alert should be issued
        """
        current_time = time.time()

        audio_config = self.config.audio_config
        cooldown_config = audio_config['cooldown']

        # Check same object cooldown
        if track_id in self.last_alert_time:
            elapsed = current_time - self.last_alert_time[track_id]
            if elapsed < cooldown_config['same_object']:
                return False

        # Check same class cooldown
        if class_name in self.last_class_alert:
            elapsed = current_time - self.last_class_alert[class_name]
            if elapsed < cooldown_config['same_class']:
                return False

        # Check any alert cooldown
        elapsed = current_time - self.last_any_alert
        if elapsed < cooldown_config['any_alert']:
            return False

        return True

    def record_alert(self, track_id: int, class_name: str):
        """Record that an alert was issued"""
        current_time = time.time()
        self.last_alert_time[track_id] = current_time
        self.last_class_alert[class_name] = current_time
        self.last_any_alert = current_time

    def create_alert(self, track, score: float, frame_width: int, frame_height: int) -> Optional[HazardAlert]:
        """
        Create hazard alert if score warrants it

        Args:
            track: Track object
            score: Hazard score
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            HazardAlert if alert should be issued, None otherwise
        """
        alert_config = self.hazard_config['alert']

        # Determine alert level
        if score >= alert_config['critical_threshold']:
            level = 'critical'
        elif score >= alert_config['warning_threshold']:
            level = 'warning'
        elif score >= alert_config['info_threshold']:
            level = 'info'
        else:
            return None  # Score too low

        # Check cooldown
        if not self.should_alert(track.track_id, track.class_name):
            return None

        # Get direction
        bbox = track.get_bbox()
        direction = self.get_direction(bbox)

        # Generate message
        message = self._generate_message(track.class_name, direction, level)

        # Create alert
        alert = HazardAlert(
            track_id=track.track_id,
            class_name=track.class_name,
            score=score,
            level=level,
            direction=direction,
            message=message,
            timestamp=time.time()
        )

        # Record alert
        self.record_alert(track.track_id, track.class_name)

        return alert

    def _generate_message(self, class_name: str, direction: str, level: str) -> str:
        """
        Generate audio alert message

        Args:
            class_name: Object class
            direction: Direction string
            level: Alert level

        Returns:
            Alert message string
        """
        audio_config = self.config.audio_config
        templates = audio_config['templates']

        template = templates.get(level, templates['warning'])

        # Format message
        message = template.format(
            class_name=class_name,
            direction=direction
        )

        return message

    def score_all_tracks(self, tracks: List, frame_width: int, frame_height: int) -> List[Dict]:
        """
        Score all tracks and return sorted by risk

        Args:
            tracks: List of Track objects
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            List of dicts with track info and scores, sorted by score (high to low)
        """
        scored_tracks = []

        for track in tracks:
            score = self.score_track(track, frame_width, frame_height)

            scored_tracks.append({
                'track': track,
                'score': score,
                'direction': self.get_direction(track.get_bbox())
            })

        # Sort by score descending
        scored_tracks.sort(key=lambda x: x['score'], reverse=True)

        return scored_tracks


if __name__ == "__main__":
    # Test hazard scorer
    config = get_config()
    scorer = HazardScorer()

    print("Hazard Scorer Initialized")
    print("\nClass Severity Map:")
    for class_name, severity in config.hazard_config['class_severity'].items():
        print(f"  {class_name:20s}: {severity:3.0f}")

    print("\nPosition Multipliers:")
    print(f"  Center: {config.hazard_config['position']['center_multiplier']}")
    print(f"  Side:   {config.hazard_config['position']['side_multiplier']}")

    print("\nSize Multipliers:")
    size_config = config.hazard_config['size']
    print(f"  Small:  {size_config['small_multiplier']}")
    print(f"  Medium: {size_config['medium_multiplier']}")
    print(f"  Large:  {size_config['large_multiplier']}")
    print(f"  XLarge: {size_config['xlarge_multiplier']}")
