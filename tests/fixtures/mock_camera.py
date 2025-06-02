import numpy as np
from unittest.mock import Mock
import threading
import time

class MockPicamera2:
    """Mock Picamera2 for testing without hardware"""

    def __init__(self):
        self.config = None
        self.controls = {}
        self.started = False
        self.frame_data = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    def create_preview_configuration(self, main=None, controls=None, transform=None):
        """Mock configuration creation"""
        return {
            'main': main or {'size': (1280, 720), 'format': 'RGB888'},
            'controls': controls or {},
            'transform': transform
        }

    def configure(self, config):
        """Mock configuration"""
        self.config = config

    def set_controls(self, controls):
        """Mock control setting"""
        self.controls.update(controls)

    def start(self):
        """Mock camera start"""
        self.started = True

    def stop(self):
        """Mock camera stop"""
        self.started = False

    def close(self):
        """Mock camera close"""
        self.started = False

    def capture_array(self, stream="main"):
        """Mock frame capture"""
        if not self.started:
            raise RuntimeError("Camera not started")

        # Simulate some capture time
        time.sleep(0.001)
        return self.frame_data.copy()

class MockCameraService:
    """Mock camera service for integration tests"""

    def __init__(self):
        self.is_initialized = True
        self.frame_count = 0
        self.mock_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    def capture_frame(self):
        self.frame_count += 1
        return self.mock_frame.copy()

    def capture_frame_rgb(self):
        return self.capture_frame()

    def capture_image_to_file(self, filename):
        return True

    def is_ready(self):
        return self.is_initialized

    def cleanup(self):
        self.is_initialized = False
