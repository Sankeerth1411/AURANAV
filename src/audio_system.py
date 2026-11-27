#!/usr/bin/env python3
"""Cross-platform offline text-to-speech alerts for AURANAV."""

import logging
import threading
from itertools import count
from queue import PriorityQueue, Empty
from typing import Optional

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import get_config

logger = logging.getLogger(__name__)


class AudioWarningSystem:
    """Serialize offline speech and let critical warnings jump the queue.

    pyttsx3 uses the operating system's installed speech engine:
    SAPI on Windows, eSpeak on many Linux systems, and NSSpeechSynthesizer
    on macOS. On Raspberry Pi OS install eSpeak (for example: espeak-ng).
    """

    def __init__(self):
        self.config = get_config()
        self.audio_config = self.config.audio_config
        self.enabled = bool(self.audio_config.get("enabled", True))
        self.rate = int(self.audio_config.get("rate", 180))
        self.volume = float(max(0.0, min(1.0, self.audio_config.get("volume", 1.0))))
        self.alert_queue = PriorityQueue()
        self._sequence = count()
        self.worker_thread: Optional[threading.Thread] = None
        self.running = False
        self.engine = None
        self._ready = threading.Event()
        self._init_error = None

        if self.enabled:
            self._start_worker()
            # Fail clearly instead of silently pretending speech is active.
            if not self._ready.wait(timeout=10):
                self.enabled = False
                logger.error("Timed out initialising offline speech engine")
            elif self._init_error is not None:
                self.enabled = False
                logger.error("Offline speech unavailable: %s", self._init_error)

    def _start_worker(self):
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker, name="AURANAV-Audio", daemon=True
        )
        self.worker_thread.start()

    def _worker(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)
        except Exception as exc:
            self._init_error = exc
            self._ready.set()
            self.running = False
            return

        self._ready.set()
        while self.running:
            try:
                priority, sequence, message = self.alert_queue.get(timeout=0.5)
            except Empty:
                continue

            try:
                if message is None:
                    break
                self.engine.say(str(message))
                self.engine.runAndWait()
                logger.info("Spoke alert: %s", message)
            except Exception:
                logger.exception("Failed to speak alert: %s", message)
            finally:
                self.alert_queue.task_done()

        try:
            if self.engine is not None:
                self.engine.stop()
        except Exception:
            logger.debug("Speech engine stop failed", exc_info=True)
        self.engine = None

    def alert(self, message: str, priority: bool = False):
        """Queue speech. Critical alerts are dequeued before normal alerts."""
        if not self.enabled or not self.running or not message:
            if message:
                logger.warning("[AUDIO UNAVAILABLE] %s", message)
            return
        queue_priority = 0 if priority else 1
        self.alert_queue.put((queue_priority, next(self._sequence), str(message)))

    def test(self):
        self.alert("Audio system test. AURANAV is ready.", priority=True)

    def shutdown(self):
        if self.worker_thread is None:
            return
        if self.running:
            self.running = False
            self.alert_queue.put((2, next(self._sequence), None))
        self.worker_thread.join(timeout=10.0)
        if self.worker_thread.is_alive():
            logger.warning("Speech worker did not stop within 10 seconds")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Test AURANAV offline speech")
    parser.add_argument("--message", default="Testing AURANAV audio system")
    parser.add_argument("--test-all", action="store_true")
    args = parser.parse_args()

    messages = (
        [
            "Pothole ahead.",
            "Vehicle approaching from the left.",
            "Obstacle on your right.",
            "Person ahead.",
            "Down stairs ahead.",
        ]
        if args.test_all else [args.message]
    )

    with AudioWarningSystem() as audio:
        for message in messages:
            audio.alert(message, priority=True)
            time.sleep(0.5)
        if audio.enabled:
            audio.alert_queue.join()


if __name__ == "__main__":
    main()
