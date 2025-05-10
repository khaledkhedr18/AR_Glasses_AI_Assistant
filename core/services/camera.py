from typing import Optional, Tuple
import cv2
import numpy as np
from picamera2 import Picamera2
import libcamera
from config.settings import Settings

class CameraService:
    """Service for handling camera operations"""

    def __init__(self):
        self.settings = Settings()
        self.picam2 = None
        self._is_initialized = False
        self.initialize()

    def initialize(self) -> bool:
        """Initialize the camera"""
        if self._is_initialized:
            return True

        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={
                    "size": (
                        self.settings.get("camera.width", 1920),
                        self.settings.get("camera.height", 1080)
                    ),
                    "format": "RGB888"
                },
                controls={
                    "AwbEnable": True,
                    "AeEnable": True,
                    "ExposureTime": 10000,
                    "AnalogueGain": 1.0,
                    "FrameDurationLimits": (33333, 33333)  # 30fps
                },
                transform=libcamera.Transform(hflip=1, vflip=1)
            )

            self.picam2.configure(config)
            controls = {"AfMode": libcamera.controls.AfModeEnum.Continuous}
            self.picam2.set_controls(controls)
            self.picam2.start()
            self._is_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            self.cleanup()
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the camera"""
        if not self._is_initialized or not self.picam2:
            print("Camera not initialized")
            return None

        try:
            frame = self.picam2.capture_array("main")
            if frame is None or frame.size == 0:
                print("Failed to capture frame")
                return None
            return frame
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def resize_frame(self, frame: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
        """Resize a frame to the specified dimensions"""
        if frame is None or frame.size == 0:
            return None

        try:
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            print(f"Error resizing frame: {e}")
            return None

    def save_frame(self, frame: np.ndarray, filename: str) -> bool:
        """Save a frame to a file"""
        if frame is None or frame.size == 0:
            print("Invalid frame data")
            return False

        try:
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return True
        except Exception as e:
            print(f"Error saving frame: {e}")
            return False

    def cleanup(self):
        """Cleanup camera resources"""
        try:
            if self.picam2:
                self.picam2.stop()
                self.picam2 = None
            self._is_initialized = False
        except Exception as e:
            print(f"Error during camera cleanup: {e}")
            self._is_initialized = False
