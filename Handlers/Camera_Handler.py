import os
import cv2
from picamera2 import Picamera2
import libcamera
from utils.Config import CameraConfig
from utils.Logging import Logger
import threading

class CameraHandler:
    # Class-level variables
    _camera_initialized = False
    _camera_init_lock = threading.Lock()
    _picam2 = None  # Shared camera instance
    _initialization_error = None

    def __init__(self):
        """
        Initialize the Camera
        """
        self.logger = Logger()
        self.logger.info("Initializing Camera Handler")
        self.camera_lock = threading.Lock()
        self.frame_buffer = None

        # Use new config structure
        self.capture_width = CameraConfig.RESOLUTION['WIDTH']
        self.capture_height = CameraConfig.RESOLUTION['HEIGHT']
        self.quality = CameraConfig.IMAGE['QUALITY']
        self.save_dir = CameraConfig.IMAGE['SAVE_DIRECTORY']
        self.saved_image_name = CameraConfig.IMAGE['FILENAME']
        self.image_save_path = None

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        with CameraHandler._camera_init_lock:
            if not CameraHandler._camera_initialized and not CameraHandler._initialization_error:
                self.__initialize_camera()
            self.picam2 = CameraHandler._picam2

    def capture_and_save_image(self):
        """
        Capture an image from the camera and save it to a static path.
        Returns: str or None: The saved image path if successful, None if failed
        """

        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Use os.path.join for cross-platform compatibility
        self.image_save_path = os.path.join(self.save_dir, self.saved_image_name)

        with self.camera_lock:
            try:
                # Capture full resolution image
                buffer = self.picam2.capture_array("main")
                # Save the image with specified quality
                cv2.imwrite(self.image_save_path, buffer, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                self.logger.info(f"Image saved as {self.image_save_path}")
                return self.image_save_path
            except Exception as e:
                self.logger.log_error_with_traceback("Error capturing image", e)
                return None

    def capture_frame(self):
        """Capture a single frame from the camera."""
        with self.camera_lock:
            try:
                if self.picam2:
                    frame = self.picam2.capture_array("main")
                    self.frame_buffer = frame  # Store latest frame
                    return frame
                return None
            except Exception as e:
                self.logger.log_error_with_traceback("Error capturing frame", e)
                return None

    def set_camera_configurations(self, exposure=None, gain=None, focus_mode=None):
        """
        set camera configurations like exposure, gain and focus mode.

        Args:
            exposure (int, optional): Exposure time in microseconds if not passed the default value will be used
            gain (float, optional): Analog gain value if not passed the default value will be used
            focus_mode (str, optional): Focus mode (auto, continuous, manual) if not passed the default value of hardware will be used

        Returns:
            bool: True if parameters were set successfully, False otherwise
        """
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
                self.logger.info("Camera configurations updated successfully")
                self.picam2.set_controls(controls)
            return True

        except Exception as e:
            self.logger.log_error_with_traceback("Error setting camera parameters", e)
            return False

    @classmethod
    def get_initialization_status(cls):
        """
        Get the camera initialization status.

        Returns:
            tuple: (bool: is_initialized, str: error_message if any)
        """
        return cls._camera_initialized, cls._initialization_error

    @classmethod
    def cleanup(cls):
        """
        Cleanup camera resources. Should be called when shutting down.
        """
        with cls._camera_init_lock:
            if cls._picam2 and cls._camera_initialized:
                try:
                    cls._picam2.stop()
                    cls._picam2.close()
                    cls._picam2 = None
                    cls._camera_initialized = False
                    cls._initialization_error = None
                except Exception as e:
                    print(f"Error during camera cleanup: {e}")

    def is_camera_connected(self):
        """
        Check if camera is properly initialized and working.

        Returns:
            bool: True if camera is working, False otherwise
        """
        if not CameraHandler._camera_initialized or CameraHandler._initialization_error:
            return False

        try:
            # Try to capture a test frame
            test_frame = self.capture_frame()
            return True
        except Exception:
            return False

    def __initialize_camera(self):
        """Initialize the camera with configured settings."""
        try:
            if not CameraHandler._picam2:
                CameraHandler._picam2 = Picamera2()

                # Configure camera using new config structure
                camera_config = {
                    "size": (self.capture_width, self.capture_height),
                    "format": CameraConfig.CONTROLS['COLOR_FORMAT']
                }

                CameraHandler._picam2.configure(**camera_config)

                # Set camera controls
                controls = {
                    "ExposureTime": CameraConfig.PARAMETERS['EXPOSURE'],
                    "AnalogueGain": CameraConfig.PARAMETERS['GAIN'],
                    "FrameDurationLimits": (CameraConfig.PARAMETERS['FRAME_DURATION'],
                                            CameraConfig.PARAMETERS['FRAME_DURATION']),
                    "FrameRate": CameraConfig.PARAMETERS['FRAME_RATE'],
                    "HFlip": CameraConfig.CONTROLS['HFLIP'],
                    "VFlip": CameraConfig.CONTROLS['VFLIP'],
                    "AeEnable": CameraConfig.CONTROLS['AE_ENABLE'],
                    "AwbEnable": CameraConfig.CONTROLS['AWB_ENABLE']
                }

                # Set focus mode
                if CameraConfig.CONTROLS['FOCUS_MODE'].lower() == "continuous":
                    controls["AfMode"] = libcamera.controls.AfModeEnum.Continuous

                CameraHandler._picam2.set_controls(controls)
                CameraHandler._picam2.start()
                CameraHandler._camera_initialized = True
                CameraHandler._initialization_error = None

        except Exception as e:
            CameraHandler._initialization_error = str(e)
            self.logger.log_error_with_traceback("Camera Initialization Error", e)
            raise