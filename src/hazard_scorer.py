#!/usr/bin/env python3
"""
AURANAV hazard prioritisation engine.

Important: image-box size is only a visual proximity cue. It is not a
metric distance, calibrated depth, or a collision-time estimate. Use a
validated depth sensor for metric distance and validate thresholds with
real users before any mobility deployment.
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
    track_id: int
    class_name: str
    score: float
    level: str
    direction: str
    message: str
    timestamp: float


class HazardScorer:
    """Rank currently observed objects by conservative visual hazard cues."""

    def __init__(self):
        self.config = get_config()
        self.hazard_config = self.config.hazard_config
        self.last_alert_time = {}
        self.last_class_alert = {}
        self.last_any_alert = 0.0

    @staticmethod
    def _safe_bbox(track):
        try:
            bbox = np.asarray(track.get_bbox(), dtype=float).reshape(4)
            if not np.all(np.isfinite(bbox)) or bbox[2] <= 0 or bbox[3] <= 0:
                return None
            return bbox
        except (AttributeError, TypeError, ValueError):
            return None

    def score_track(self, track, frame_width: int, frame_height: int) -> float:
        bbox = self._safe_bbox(track)
        if bbox is None:
            return 0.0

        severity = self._get_class_severity(track.class_name)
        confidence_values = getattr(track, "confidence_history", [])
        confidence = float(np.mean(confidence_values[-5:])) if confidence_values else 0.5
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Normalized visual cues only. No metric distance is inferred.
        cx, cy, width, height = bbox
        area = float(np.clip(width * height, 0.0, 1.0))
        size_factor = self._get_size_multiplier(bbox)
        position_factor = self._get_position_multiplier(bbox, frame_width, frame_height)
        bottom_factor = 1.0 + 0.25 * float(np.clip((cy + height / 2) - 0.55, 0.0, 0.45)) / 0.45
        movement_factor = self._get_movement_multiplier(track)

        # Combine factors, then clamp. A high score means "prioritise this cue",
        # not a calibrated probability of collision.
        score = severity * position_factor * size_factor * bottom_factor * movement_factor
        score *= 0.65 + 0.35 * confidence
        return float(np.clip(score, 0.0, 100.0))

    def _get_class_severity(self, class_name: str) -> float:
        severity_map = self.hazard_config.get("class_severity", {})
        name = str(class_name).strip().lower()
        if name in severity_map:
            return float(severity_map[name])
        if name in {"car", "truck", "bus", "van", "vehicle"}:
            return float(severity_map.get("vehicle", 80))
        # Unknown classes should be noticed, but not automatically treated as
        # an imminent danger. Their risk also depends on position and size.
        return 45.0

    def _get_position_multiplier(self, bbox, frame_width: int, frame_height: int) -> float:
        cx = float(bbox[0])
        cfg = self.hazard_config.get("position", {})
        center_width = float(cfg.get("center_zone_width", 0.4))
        center_mult = float(cfg.get("center_multiplier", 1.4))
        side_mult = float(cfg.get("side_multiplier", 0.8))
        left, right = 0.5 - center_width / 2, 0.5 + center_width / 2
        return center_mult if left <= cx <= right else side_mult

    def _get_size_multiplier(self, bbox) -> float:
        area = float(np.clip(bbox[2] * bbox[3], 0.0, 1.0))
        cfg = self.hazard_config.get("size", {})
        if area >= float(cfg.get("large_threshold", 0.30)):
            return float(cfg.get("xlarge_multiplier", 2.0))
        if area >= float(cfg.get("medium_threshold", 0.15)):
            return float(cfg.get("large_multiplier", 1.5))
        if area >= float(cfg.get("small_threshold", 0.05)):
            return float(cfg.get("medium_multiplier", 1.0))
        return float(cfg.get("small_multiplier", 0.5))

    def _get_movement_multiplier(self, track) -> float:
        cfg = self.hazard_config.get("movement", {})
        rate = 0.0
        try:
            rate = float(track.get_size_change_rate())
        except (AttributeError, TypeError, ValueError):
            pass
        threshold = float(cfg.get("velocity_threshold", 0.02))
        if rate > threshold:
            return float(cfg.get("approaching_multiplier", 1.4))
        if rate < -threshold:
            return float(cfg.get("receding_multiplier", 0.8))
        return float(cfg.get("stationary_multiplier", 1.0))

    def get_direction(self, bbox) -> str:
        cx = float(bbox[0])
        cfg = self.config.audio_config.get("directions", {})
        left = float(cfg.get("left_threshold", 0.33))
        right = float(cfg.get("right_threshold", 0.67))
        if cx < left:
            return str(cfg.get("left_label", "left"))
        if cx > right:
            return str(cfg.get("right_label", "right"))
        return str(cfg.get("center_label", "ahead"))

    def should_alert(self, track_id: int, class_name: str, level: str = "warning") -> bool:
        now = time.monotonic()
        cooldown = self.config.audio_config.get("cooldown", {})
        track_wait = float(cooldown.get("same_object", 3.0))
        class_wait = float(cooldown.get("same_class", 2.0))
        global_wait = float(cooldown.get("any_alert", 0.7))

        if now - self.last_alert_time.get(track_id, 0.0) < track_wait:
            return False
        if now - self.last_class_alert.get(class_name, 0.0) < class_wait:
            return False
        # Critical alerts are allowed to bypass the global throttle so a lower
        # priority message cannot suppress an urgent stair/vehicle warning.
        if level != "critical" and now - self.last_any_alert < global_wait:
            return False
        return True

    def record_alert(self, track_id: int, class_name: str):
        now = time.monotonic()
        self.last_alert_time[track_id] = now
        self.last_class_alert[class_name] = now
        self.last_any_alert = now

    def create_alert(self, track, score: float, frame_width: int, frame_height: int) -> Optional[HazardAlert]:
        thresholds = self.hazard_config.get("alert", {})
        if score >= float(thresholds.get("critical_threshold", 70)):
            level = "critical"
        elif score >= float(thresholds.get("warning_threshold", 50)):
            level = "warning"
        elif score >= float(thresholds.get("info_threshold", 30)):
            level = "info"
        else:
            return None

        if not self.should_alert(track.track_id, track.class_name, level):
            return None

        bbox = self._safe_bbox(track)
        if bbox is None:
            return None
        direction = self.get_direction(bbox)
        templates = self.config.audio_config.get("templates", {})
        template = templates.get(level, "{class_name} on your {direction}")
        message = template.format(class_name=track.class_name, direction=direction)

        alert = HazardAlert(
            track_id=int(track.track_id),
            class_name=str(track.class_name),
            score=float(score),
            level=level,
            direction=direction,
            message=message,
            timestamp=time.time(),
        )
        self.record_alert(track.track_id, track.class_name)
        return alert

    def score_all_tracks(self, tracks: List, frame_width: int, frame_height: int) -> List[Dict]:
        scored = []
        for track in tracks:
            score = self.score_track(track, frame_width, frame_height)
            scored.append({
                "track": track,
                "score": score,
                "direction": self.get_direction(track.get_bbox()),
            })
        return sorted(scored, key=lambda item: item["score"], reverse=True)


if __name__ == "__main__":
    config = get_config()
    scorer = HazardScorer()
    print("AURANAV hazard prioritisation ready")
    print("Scores are relative visual priorities, not collision probabilities or distances.")
    for name, severity in config.hazard_config.get("class_severity", {}).items():
        print(f"  {name:20s}: {severity:3.0f}")
