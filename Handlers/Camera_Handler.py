import os
import cv2
from picamera2 import Picamera2
import libcamera
from utils.Config import CAMERA_CONFIG
from utils.Logging import Logger
import threading


class CameraHandler:
    """Handles camera operations with simplified access control and frame buffering."""

    # Class-level variables for shared resources
    _camera_init_lock = threading.Lock()  # Lock for camera initialization
    _camera_initialized = False  # Flag to track if camera is initialized
    _camera_busy_lock = threading.Lock()  # Lock for camera busy state
    _picam2 = None  # Shared camera instance
    _initialization_error = None
    _camera_busy = False  # Shared camera busy state

    # Class-level event for tracking initialization completion
    _initialization_complete = threading.Event()

    def __init__(self):
        """Initialize Camera Handler with configurations and frame buffer."""
        self.logger = Logger()
        self.logger.info("Initializing Camera Handler")

        # Instance-specific attributes
        self.frame_buffer = None

        # Load configuration
        self.__load_config()

        # Initialize camera only for first instance
        with self._camera_init_lock:
            if not self._camera_initialized:
                try:
                    self.__initialize_camera()
                    self.__class__._camera_initialized = True
                    self._initialization_complete.set()
                    self.logger.info("Initial camera initialization completed successfully")
                except Exception as e:
                    self.logger.error(f"Camera initialization failed: {str(e)}")
                    self._initialization_complete.clear()
                    self._initialization_error = str(e)
                    raise

        # Use shared camera instance
        self.picam2 = self._picam2

    def capture_frame(self):
        """
        Capture a frame from camera or return buffered frame if camera is busy.

        Returns:
            numpy.ndarray or None: Captured/buffered frame or None if failed
        """
        if not self._initialization_complete.wait(timeout=5):  # 5 seconds timeout
            self.logger.error("Camera initialization timeout exceeded")
            return None

        # Return buffered frame if camera is busy
        with self._camera_busy_lock:
            if self._camera_busy:
                return self.frame_buffer

            try:
                self._camera_busy = True
                if self.picam2:
                    frame = self.picam2.capture_array("main")
                    self.frame_buffer = frame  # Update frame buffer
                    return frame
                return None
            except Exception as e:
                self.logger.log_error_with_traceback("Error capturing frame", e)
                return None
            finally:
                self._camera_busy = False

    def capture_and_save_image(self):
        """
        Capture and save image, using buffered frame if camera is busy.

        Returns:
            str or None: Path to saved image or None if failed
        """
        if not self._initialization_complete.wait(timeout=5):
            self.logger.error("Camera initialization timeout exceeded")
            return None

        # Use buffered frame if camera is busy
        with self._camera_busy_lock:
            if self._camera_busy and self.frame_buffer is not None:
                try:
                    cv2.imwrite(self.image_save_path, self.frame_buffer,
                                [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    return self.image_save_path
                except Exception as e:
                    self.logger.log_error_with_traceback("Error saving buffered image", e)
                    return None

            try:
                self._camera_busy = True
                if self.picam2:
                    frame = self.picam2.capture_array("main")
                    self.frame_buffer = frame  # Update frame buffer

                    cv2.imwrite(self.image_save_path, frame,
                                [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    return self.image_save_path
                return None
            except Exception as e:
                self.logger.log_error_with_traceback("Error capturing and saving image", e)
                return None
            finally:
                self._camera_busy = False

    def set_camera_configurations(self, exposure=None, gain=None, focus_mode=None):
        """
        Set camera configurations.

        Args:
            exposure (int, optional): Exposure time in microseconds
            gain (float, optional): Analog gain value
            focus_mode (str, optional): Focus mode (auto, continuous, manual)

        Returns:
            bool: True if successful, False otherwise
        """
        if not self._initialization_complete.wait(timeout=5):
            self.logger.error("Camera initialization timeout exceeded")
            return False

        if not self.picam2:
            self.logger.warning("Camera not initialized, cannot set configurations")
            return False

        try:
            controls = {}
            if exposure is not None:
                controls["ExposureTime"] = exposure
            if gain is not None:
                controls["AnalogueGain"] = gain
            if focus_mode is not None:
                if focus_mode.lower() == "auto":
                    controls["AfMode"] = libcamera.controls.AfModeEnum.Auto
                elif focus_mode.lower() == "continuous":
                    controls["AfMode"] = libcamera.controls.AfModeEnum.Continuous
                elif focus_mode.lower() == "manual":
                    controls["AfMode"] = libcamera.controls.AfModeEnum.Manual

            if controls:
                with self._camera_busy_lock:
                    self.picam2.set_controls(controls)
                self.logger.info("Camera configurations updated successfully")
            return True
        except Exception as e:
            self.logger.log_error_with_traceback("Error setting camera parameters", e)
            return False

    @classmethod
    def cleanup(cls):
        """Clean up camera resources when shutting down."""
        with cls._camera_init_lock:
            if cls._picam2 and cls._camera_initialized:
                try:
                    cls._picam2.stop()
                    cls._picam2.close()
                    cls._picam2 = None
                    cls._camera_initialized = False
                    cls._initialization_error = None
                    cls._initialization_complete.clear()
                except Exception as e:
                    print(f"Error during camera cleanup: {e}")

    def __initialize_camera(self):
        """Initialize the camera with configured settings."""
        try:
            if not self._picam2:
                self._picam2 = Picamera2()

                # Create a camera configuration first with correct structure
                # The size parameter should be inside the main dictionary, not a separate parameter
                preview_config = self._picam2.create_preview_configuration(
                    main={
                        "format": CAMERA_CONFIG.get('CONTROLS', {}).get('COLOR_FORMAT', 'RGB888'),
                        "size": (self.capture_width, self.capture_height)  # Put size inside the main dict
                    }
                )

                # Apply the configuration
                self._picam2.configure(preview_config)

                # Get parameters and controls from config
                params = CAMERA_CONFIG.get('PARAMETERS', {})
                controls = CAMERA_CONFIG.get('CONTROLS', {})

                # Set camera controls from config with defaults
                control_settings = {
                    "ExposureTime": params.get('EXPOSURE', 10000),
                    "AnalogueGain": params.get('GAIN', 1.0),
                    "FrameDurationLimits": (
                        params.get('FRAME_DURATION', 33333),
                        params.get('FRAME_DURATION', 33333)
                    ),
                    "FrameRate": params.get('FRAME_RATE', 30),
                    "AeEnable": controls.get('AE_ENABLE', True),
                    "AwbEnable": controls.get('AWB_ENABLE', True)
                }

                # Set focus mode if specified
                if controls.get('FOCUS_MODE', '').lower() == "continuous":
                    control_settings["AfMode"] = libcamera.controls.AfModeEnum.Continuous

                self._picam2.set_controls(control_settings)
                self._picam2.start()

        except Exception as e:
            self._initialization_error = str(e)
            self.logger.log_error_with_traceback("Camera Initialization Error", e)
            raise

    def __load_config(self):
        """Load configuration settings for camera."""
        # Load resolution settings with defaults
        resolution = CAMERA_CONFIG.get('RESOLUTION', {})
        self.capture_width = resolution.get('WIDTH', 1920)
        self.capture_height = resolution.get('HEIGHT', 1080)

        # Load image settings with defaults
        image_config = CAMERA_CONFIG.get('IMAGE', {})
        self.quality = image_config.get('QUALITY', 90)
        self.save_dir = image_config.get('SAVE_DIRECTORY', '/tmp/AIAssistant/')
        self.saved_image_name = image_config.get('FILENAME', 'captured_image.jpg')

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        self.image_save_path = os.path.join(self.save_dir, self.saved_image_name)
