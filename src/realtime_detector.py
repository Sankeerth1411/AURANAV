#!/usr/bin/env python3
"""
AURANAV - Real-Time Detection System
Live camera object detection with hazard warnings
"""

import sys
import cv2
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import get_config
from src.object_tracker import ObjectTracker
from src.hazard_scorer import HazardScorer
from src.audio_system import AudioWarningSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeDetector:
    """Real-time object detection and hazard warning system"""

    def __init__(self, weights_path: str, camera_source=0):
        """
        Initialize detector

        Args:
            weights_path: Path to trained YOLO weights
            camera_source: Camera index (0 for default) or video file path
        """
        self.config = get_config()
        self.weights_path = Path(weights_path)
        self.camera_source = camera_source

        # Load model
        logger.info(f"Loading model from {self.weights_path}")
        self.model = self._load_model()
        self.class_names = self.model.names

        # Optional secondary discovery model. It is deliberately not connected
        # to tracking or spoken hazard alerts until validated independently.
        open_vocab_config = self.config.get('open_vocabulary', default={}) or {}
        ov_device = open_vocab_config.get('device', self.config.inference_config.get('device', 'cpu'))
        if ov_device == 'auto':
            import torch
            ov_device = '0' if torch.cuda.is_available() else 'cpu'
        from src.open_vocabulary import OpenVocabularyDiscovery
        self.open_vocabulary = OpenVocabularyDiscovery(open_vocab_config, device=ov_device)
        self.discovery_interval = max(1, int(open_vocab_config.get('every_n_frames', 10)))
        self.discovery_frame_counter = 0

        # Initialize components
        self.tracker = self._init_tracker()
        self.hazard_scorer = HazardScorer()
        self.audio_system = AudioWarningSystem()

        # Camera
        self.camera = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0

        # Statistics
        self.frame_count = 0
        self.start_time = None
        self.inference_times = []

    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        model = YOLO(str(self.weights_path))

        # Set device
        inference_config = self.config.inference_config
        device = inference_config.get('device', 'auto')

        if device == 'auto':
            import torch
            device = '0' if torch.cuda.is_available() else 'cpu'

        model.to(device)

        logger.info(f"Model loaded successfully (device: {device})")
        logger.info(f"Classes: {len(model.names)}")

        return model

    def _init_tracker(self) -> ObjectTracker:
        """Initialize object tracker"""
        tracking_config = self.config.tracking_config

        if not tracking_config.get('enabled', True):
            logger.info("Object tracking disabled")
            return None

        tracker = ObjectTracker(
            max_age=tracking_config.get('max_age', 30),
            min_hits=tracking_config.get('min_hits', 3),
            iou_threshold=tracking_config.get('iou_threshold', 0.3)
        )

        logger.info("Object tracker initialized")
        return tracker

    def open_camera(self) -> bool:
        """
        Open camera connection

        Returns:
            True if successful
        """
        camera_config = self.config.camera_config

        # Open a video file directly; use DirectShow only for camera indices.
        if isinstance(self.camera_source, (str, Path)):
            self.camera = cv2.VideoCapture(str(self.camera_source))
        else:
            self.camera = cv2.VideoCapture(
                self.camera_source, cv2.CAP_DSHOW
        )
        if not self.camera.isOpened():
            logger.error(f"Failed to open camera: {self.camera_source}")
            return False

        # Set camera properties
        width = camera_config.get('width', 640)
        height = camera_config.get('height', 480)
        fps = camera_config.get('fps', 30)
        buffer_size = camera_config.get('buffer_size', 1)

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.CAP_PROP_FPS, fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)

        # Get actual properties
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.camera.get(cv2.CAP_PROP_FPS)

        logger.info(f"Camera opened: {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
        return True

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame

        Args:
            frame: Input frame (BGR)

        Returns:
            Dictionary with detection results
        """
        # Run inference
        inference_start = time.time()

        inference_config = self.config.inference_config
        results = self.model(
            frame,
            conf=inference_config.get('confidence_threshold', 0.25),
            iou=inference_config.get('iou_threshold', 0.45),
            max_det=inference_config.get('max_detections', 100),
            verbose=False
        )[0]

        inference_time = time.time() - inference_start

        # Convert primary detector outputs to normalized format
        detections = self._parse_detections(results)

        # Run optional open-vocabulary discovery intermittently to limit CPU load.
        # These candidates are shown separately and do not trigger audio alerts.
        self.discovery_frame_counter += 1
        discovery_detections = []
        if (
            self.open_vocabulary.available
            and self.discovery_frame_counter % self.discovery_interval == 0
        ):
            discovery_detections = self.open_vocabulary.predict(frame, detections)

        # Update tracker
        if self.tracker:
            tracks = self.tracker.update(detections)
        else:
            # No tracking, treat each detection as a track
            tracks = [self._detection_to_track(d, i) for i, d in enumerate(detections)]

        # Score hazards
        scored_tracks = self.hazard_scorer.score_all_tracks(
            tracks,
            self.frame_width,
            self.frame_height
        )

        # Generate alerts
        alerts = []
        for scored in scored_tracks:
            track = scored['track']
            score = scored['score']

            alert = self.hazard_scorer.create_alert(
                track, score, self.frame_width, self.frame_height
            )

            if alert:
                alerts.append(alert)
                # Send to audio system
                self.audio_system.alert(alert.message, priority=(alert.level == 'critical'))

        processing_time = time.time() - inference_start
        self.inference_times.append(processing_time)

        return {
            'detections': detections,
            'tracks': tracks,
            'scored_tracks': scored_tracks,
            'alerts': alerts,
            'discovery_detections': discovery_detections,
            'inference_time': inference_time,
            'processing_time': processing_time
        }

    def _parse_detections(self, results) -> List[Dict]:
        """Parse YOLO results into detection dictionaries"""
        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box

            # Convert to normalized center format [x, y, w, h]
            x_center = ((x1 + x2) / 2) / self.frame_width
            y_center = ((y1 + y2) / 2) / self.frame_height
            width = (x2 - x1) / self.frame_width
            height = (y2 - y1) / self.frame_height

            detections.append({
                'bbox': [x_center, y_center, width, height],
                'bbox_pixels': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class_id': int(cls_id),
                'class_name': self.class_names[cls_id]
            })

        return detections

    def _detection_to_track(self, detection: Dict, track_id: int):
        """Convert detection to simple track object for no-tracking mode"""
        class SimpleTrack:
            def __init__(self, det, tid):
                self.track_id = tid
                self.class_id = det['class_id']
                self.class_name = det['class_name']
                self.bbox_history = [det['bbox']]
                self.confidence_history = [det['confidence']]

            def get_bbox(self):
                return self.bbox_history[-1]

            def get_size_change_rate(self):
                return 0.0

        return SimpleTrack(detection, track_id)

    def draw_results(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detection results on frame

        Args:
            frame: Input frame
            results: Detection results

        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()

        # Draw tracks
        for scored in results['scored_tracks']:
            track = scored['track']
            score = scored['score']
            direction = scored['direction']

            
            # Get normalized bbox and convert to pixel coordinates
            bbox = track.get_bbox()
            x, y, w, h = map(float, bbox)

            # Skip invalid Kalman-filter predictions
            if not np.all(np.isfinite([x, y, w, h])) or w <= 0 or h <= 0:
                continue

            # Convert normalized center coordinates to pixel corners
            x1 = int(np.clip((x - w / 2) * self.frame_width, 0, self.frame_width - 1))
            y1 = int(np.clip((y - h / 2) * self.frame_height, 0, self.frame_height - 1))
            x2 = int(np.clip((x + w / 2) * self.frame_width, 0, self.frame_width - 1))
            y2 = int(np.clip((y + h / 2) * self.frame_height, 0, self.frame_height - 1))

            # Skip boxes that are too small or invalid
            if x2 <= x1 or y2 <= y1:
                continue

            # Color based on risk
            if score >= 70:
                color = (0, 0, 255)  # Red
            elif score >= 50:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green

            # Draw bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"#{track.track_id} {track.class_name} {score:.0f}"
            cv2.putText(
                vis_frame, label,
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

        # Draw discovery candidates distinctly. They are informational only:
        # no tracker state, hazard score, or spoken warning is generated.
        for candidate in results.get('discovery_detections', []):
            x1, y1, x2, y2 = candidate['bbox_pixels']
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            label = f"DISCOVERY {candidate['class_name']} {candidate['confidence']:.2f}"
            cv2.putText(
                vis_frame, label, (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
            )

        # Draw stats
        self._draw_stats(vis_frame, results)

        return vis_frame

    def _draw_stats(self, frame: np.ndarray, results: Dict):
        """Draw statistics overlay"""
        # FPS
        if len(self.inference_times) > 0:
            avg_inference = np.mean(self.inference_times[-30:])
            fps = 1.0 / avg_inference if avg_inference > 0 else 0
        else:
            fps = 0

        # Stats text
        stats = [
            f"FPS: {fps:.1f}",
            f"Objects: {len(results['tracks'])}",
            f"Alerts: {len(results['alerts'])}"
        ]

        y_offset = 30
        for stat in stats:
            cv2.putText(
                frame, stat,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )
            y_offset += 30

    def run(self, display: bool = True, max_frames: Optional[int] = None):
        """
        Run detection loop

        Args:
            display: Show live visualization
            max_frames: Maximum frames to process (None for infinite)
        """
        if not self.open_camera():
            return

        logger.info("Starting detection loop (press 'q' to quit)")
        self.start_time = time.time()

        try:
            while True:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break

                self.frame_count += 1

                # Process frame
                results = self.process_frame(frame)
                # Draw visualization. Keep the original frame for inference;
                # resize only the annotated copy shown on screen.
                if display:
                    vis_frame = self.draw_results(frame, results)
                    max_width, max_height = 1280, 720
                    h, w = vis_frame.shape[:2]
                    scale = min(max_width / w, max_height / h, 1.0)

                    if scale < 1.0:
                        display_frame = cv2.resize(
                            vis_frame,
                            (max(1, int(w * scale)), max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA
                        )
                    else:
                        display_frame = vis_frame

                    cv2.imshow('AURANAV - Real-Time Detection', display_frame)

                    # Handle key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("User requested quit")
                        break

                # Check max frames
                if max_frames and self.frame_count >= max_frames:
                    logger.info(f"Reached max frames: {max_frames}")
                    break

                # Log progress periodically
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    avg_fps = self.frame_count / elapsed
                    logger.info(f"Processed {self.frame_count} frames (avg {avg_fps:.1f} FPS)")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up...")

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()

        if self.audio_system:
            self.audio_system.shutdown()

        # Print final statistics
        if self.frame_count > 0:
            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed
            avg_inference = np.mean(self.inference_times)

            logger.info("=" * 60)
            logger.info("SESSION STATISTICS")
            logger.info("=" * 60)
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Total time: {elapsed:.2f} seconds")
            logger.info(f"Average FPS: {avg_fps:.2f}")
            logger.info(f"Average inference time: {avg_inference*1000:.2f} ms")
            logger.info("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AURANAV Real-Time Detection System'
    )
    parser.add_argument(
        '--weights',
        type=str,
        help='Path to model weights (default: from config)'
    )
    parser.add_argument(
        '--camera',
        type=str,
        default='0',
        help='Camera source (0 for default camera, or video file path)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable visual display (audio only)'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum frames to process (default: infinite)'
    )
    args = parser.parse_args()

    try:
        # Get weights path
        if args.weights:
            weights_path = args.weights
        else:
            config = get_config()
            weights_path = config.get_path('weights') / 'auranav_best.pt'

        if not Path(weights_path).exists():
            logger.error(f"Model weights not found: {weights_path}")
            logger.info("Train the model first using train_model.py")
            sys.exit(1)

        # Parse camera source
        camera_source = args.camera
        if camera_source.isdigit():
            camera_source = int(camera_source)

        # Initialize detector
        detector = RealtimeDetector(weights_path, camera_source)

        # Run detection
        detector.run(
            display=not args.no_display,
            max_frames=args.max_frames
        )

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
