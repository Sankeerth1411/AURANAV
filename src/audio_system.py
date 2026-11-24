
#!/usr/bin/env python3
"""AURANAV - Windows native text-to-speech warning system."""

import sys
import logging
import threading
from pathlib import Path
from queue import Queue, Empty

sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AudioWarningSystem:
    """Text-to-speech alerts using Windows SAPI on Windows."""

    def __init__(self):
        self.config = get_config()
        self.audio_config = self.config.audio_config
        self.enabled = self.audio_config.get("enabled", True)
        self.engine_type = "sapi"

        self.alert_queue = Queue()
        self.worker_thread = None
        self.running = False
        self.engine = None
        self.speaker_lock = threading.Lock()

        if self.enabled:
            self._init_engine()
            if self.enabled:
                self._start_worker()
        else:
            logger.info("Audio warnings disabled in config")

    def _init_engine(self):
        """Initialize Windows speech synthesis."""
        if sys.platform != "win32":
            logger.error("This SAPI audio backend requires Windows.")
            self.enabled = False
            return

        try:
            import win32com.client
            import pythoncom

            # Store modules for use by the speech worker thread.
            self.pythoncom = pythoncom
            self.win32com = win32com.client

            # Probe that Windows speech synthesis is available.
            self.pythoncom.CoInitialize()
            try:
                speaker = self.win32com.Dispatch("SAPI.SpVoice")
                speaker.Volume = int(
                    max(0, min(100, self.audio_config.get("volume", 1.0) * 100))
                )
                speaker.Rate = 0
                del speaker
            finally:
                self.pythoncom.CoUninitialize()

            logger.info("Audio engine initialized: Windows SAPI")
        except Exception:
            logger.exception("Could not initialize Windows SAPI speech")
            self.enabled = False

    def _start_worker(self):
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker, name="AURANAV-Audio", daemon=True
        )
        self.worker_thread.start()
        logger.info("Audio worker thread started")

    def _worker(self):
        """Create and use the SAPI COM object on this worker thread."""
        initialized_com = False
        speaker = None

        try:
            self.pythoncom.CoInitialize()
            initialized_com = True
            speaker = self.win32com.Dispatch("SAPI.SpVoice")
            speaker.Volume = int(
                max(0, min(100, self.audio_config.get("volume", 1.0) * 100))
            )
            speaker.Rate = 0

            while self.running:
                try:
                    message = self.alert_queue.get(timeout=0.5)
                except Empty:
                    continue

                if message is None:
                    self.alert_queue.task_done()
                    break

                try:
                    speaker.Speak(str(message))
                    logger.info("Spoke: '%s'", message)
                except Exception:
                    logger.exception("Failed to speak message: %s", message)
                finally:
                    self.alert_queue.task_done()

        except Exception:
            logger.exception("Windows speech worker failed")
        finally:
            speaker = None
            if initialized_com:
                self.pythoncom.CoUninitialize()

    def _speak(self, message: str):
        """Queue speech so all messages use the same worker thread."""
        if self.enabled and self.running:
            self.alert_queue.put(message)

    def alert(self, message: str, priority: bool = False):
        """Queue a spoken warning; priority is retained for API compatibility."""
        if not self.enabled:
            logger.warning("[AUDIO DISABLED] %s", message)
            return

        # Serialize speech through the worker to avoid competing speech engines.
        self._speak(message)

    def test(self):
        self.alert("Audio system test. AURANAV is ready.")

    def shutdown(self):
        logger.info("Shutting down audio system...")
        if self.running:
            self.running = False
            self.alert_queue.put(None)

        if self.worker_thread:
            self.worker_thread.join(timeout=10.0)
            if self.worker_thread.is_alive():
                logger.warning("Audio worker did not stop within 10 seconds")

        logger.info("Audio system shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


def main():
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Test AURANAV audio system")
    parser.add_argument("--message", default="Testing AURANAV audio system")
    parser.add_argument("--test-all", action="store_true")
    args = parser.parse_args()

    messages = (
        [
            "Test one. Pothole ahead.",
            "Test two. Vehicle approaching from the left.",
            "Test three. Obstacle close on your right.",
            "Test four. Person ahead.",
            "Test five. Down stairs ahead.",
        ]
        if args.test_all
        else [args.message]
    )

    with AudioWarningSystem() as audio:
        for message in messages:
            logger.info("Testing: %s", message)
            audio.alert(message)
            time.sleep(0.5)

        # Allow queued messages to finish before shutting down.
        audio.alert_queue.join()


if __name__ == "__main__":
    main()
