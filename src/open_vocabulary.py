#!/usr/bin/env python3
"""
Optional open-vocabulary object discovery for AURANAV.

This is a secondary discovery channel, not a safety-rated detector. Its
outputs are intentionally kept separate from the tracker and audio hazard
pipeline until the model has been validated on the deployment dataset.
"""

import logging
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def box_iou_xywh(a, b) -> float:
    """IoU for normalized [center_x, center_y, width, height] boxes."""
    a = np.asarray(a, dtype=float).reshape(4)
    b = np.asarray(b, dtype=float).reshape(4)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return 0.0
    if np.any(a[2:] <= 0) or np.any(b[2:] <= 0):
        return 0.0

    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2

    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return float(intersection / union) if union > 1e-12 else 0.0


def remove_primary_duplicates(
    candidates: List[Dict], primary: List[Dict], iou_threshold: float = 0.55
) -> List[Dict]:
    """Suppress open-vocabulary boxes already covered by the primary model."""
    kept = []
    for candidate in candidates:
        candidate_box = candidate.get("bbox")
        if candidate_box is None:
            continue
        duplicate = any(
            box_iou_xywh(candidate_box, detection.get("bbox", [])) >= iou_threshold
            for detection in primary
        )
        if not duplicate:
            kept.append(candidate)
    return kept


class OpenVocabularyDiscovery:
    """Lazy, optional YOLO-World wrapper; gracefully unavailable if unsupported."""

    def __init__(self, config: Dict, device: str = "cpu"):
        self.enabled = bool(config.get("enabled", False))
        self.available = False
        self.model = None
        self.classes = [str(x).strip() for x in config.get("classes", []) if str(x).strip()]
        self.confidence = float(config.get("confidence_threshold", 0.25))
        self.image_size = int(config.get("image_size", 416))
        self.max_detections = int(config.get("max_detections", 30))
        self.iou_threshold = float(config.get("primary_dedup_iou", 0.55))

        if not self.enabled:
            logger.info("Open-vocabulary discovery disabled")
            return
        if not self.classes:
            logger.warning("Open-vocabulary discovery enabled but no classes configured")
            return

        weights = str(config.get("weights", "yolov8s-worldv2.pt"))
        try:
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(weights)
            self.model.set_classes(self.classes)
            self.model.to(device)
            self.available = True
            logger.info(
                "Open-vocabulary discovery ready (%s; %d prompts; device=%s)",
                weights, len(self.classes), device
            )
        except Exception as exc:
            self.model = None
            self.available = False
            logger.warning("Open-vocabulary discovery unavailable: %s", exc)

    def predict(self, frame, primary_detections=None) -> List[Dict]:
        """Return novel candidate objects; never sends alerts or changes tracks."""
        if not self.available or self.model is None:
            return []

        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            return []

        try:
            result = self.model.predict(
                frame,
                conf=self.confidence,
                imgsz=self.image_size,
                max_det=self.max_detections,
                verbose=False,
            )[0]
            if result.boxes is None or len(result.boxes) == 0:
                return []

            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            names = result.names
            candidates = []

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(float, box)
                x1, x2 = np.clip([x1, x2], 0, width)
                y1, y2 = np.clip([y1, y2], 0, height)
                if x2 <= x1 or y2 <= y1:
                    continue

                if isinstance(names, dict):
                    class_name = str(names.get(int(class_id), f"object_{class_id}"))
                else:
                    class_name = str(names[int(class_id)])

                candidates.append({
                    "bbox": [
                        ((x1 + x2) / 2) / width,
                        ((y1 + y2) / 2) / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    ],
                    "bbox_pixels": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "source": "open_vocabulary",
                })

            return remove_primary_duplicates(
                candidates, primary_detections or [], self.iou_threshold
            )
        except Exception:
            logger.exception("Open-vocabulary inference failed; primary detector continues")
            return []
