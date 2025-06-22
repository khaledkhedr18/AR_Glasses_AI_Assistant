import os
import cv2
from picamera2 import Picamera2
import libcamera
from utils.Config import CameraConfig
from utils.Logging import Logger


class CameraHandler:
    """Handles camera operations with simplified access control and frame buffering."""

    # Class-level variables for singleton camera instance
    _camera_initialized = False
    _picam2 = None  # Shared camera instance
    _initialization_error = None
    _camera_busy = False  # Class-level camera busy state

    def __init__(self):
        """Initialize Camera Handler with configurations and frame buffer."""
        self.logger = Logger()
        self.logger.info("Initializing Camera Handler")

        # Frame buffer for busy state handling
        self.frame_buffer = None

        # Configure camera settings from CameraConfig
        self.capture_width = CameraConfig.RESOLUTION['WIDTH']
        self.capture_height = CameraConfig.RESOLUTION['HEIGHT']
        self.quality = CameraConfig.IMAGE['QUALITY']
        self.save_dir = CameraConfig.IMAGE['SAVE_DIRECTORY']
        self.saved_image_name = CameraConfig.IMAGE['FILENAME']
        self.image_save_path = None

        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize camera if not already initialized
        if not CameraHandler._camera_initialized and not CameraHandler._initialization_error:
            self.__initialize_camera()
        self.picam2 = CameraHandler._picam2

    def capture_frame(self):
        """
        Capture a frame from camera or return buffered frame if camera is busy.

        Returns:
            numpy.ndarray or None: Captured/buffered frame or None if failed
        """
        # Return buffered frame if camera is busy
        if CameraHandler._camera_busy:
            return self.frame_buffer

        try:
            CameraHandler._camera_busy = True
            if self.picam2:
                frame = self.picam2.capture_array("main")
                self.frame_buffer = frame  # Update frame buffer
                return frame
            return None
        except Exception as e:
            self.logger.log_error_with_traceback("Error capturing frame", e)
            return None
        finally:
            CameraHandler._camera_busy = False

    def capture_and_save_image(self):
        """
        Capture and save image, using buffered frame if camera is busy.

        Returns:
            str or None: Path to saved image or None if failed
        """
        # Use buffered frame if camera is busy
        if CameraHandler._camera_busy and self.frame_buffer is not None:
            try:
                self.image_save_path = os.path.join(self.save_dir, self.saved_image_name)
                cv2.imwrite(self.image_save_path, self.frame_buffer,
                            [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                return self.image_save_path
            except Exception as e:
                self.logger.log_error_with_traceback("Error saving buffered image", e)
                return None

        try:
            CameraHandler._camera_busy = True
            if self.picam2:
                frame = self.picam2.capture_array("main")
                self.frame_buffer = frame  # Update frame buffer
                self.image_save_path = os.path.join(self.save_dir, self.saved_image_name)
                cv2.imwrite(self.image_save_path, frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                return self.image_save_path
            return None
        except Exception as e:
            self.logger.log_error_with_traceback("Error capturing and saving image", e)
            return None
        finally:
            CameraHandler._camera_busy = False

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
                self.picam2.set_controls(controls)
                self.logger.info("Camera configurations updated successfully")
            return True
        except Exception as e:
            self.logger.log_error_with_traceback("Error setting camera parameters", e)
            return False

    @classmethod
    def cleanup(cls):
        """Clean up camera resources when shutting down."""
        if cls._picam2 and cls._camera_initialized:
            try:
                cls._picam2.stop()
                cls._picam2.close()
                cls._picam2 = None
                cls._camera_initialized = False
                cls._initialization_error = None
            except Exception as e:
                print(f"Error during camera cleanup: {e}")

    def is_camera_properly_working(self):
        """
        Check if camera is properly initialized and working.

        Returns:
            bool: True if camera is working properly, False otherwise
        """
        if not CameraHandler._camera_initialized or CameraHandler._initialization_error:
            return False

        test_frame = self.capture_frame()
        if test_frame is not None:
            return True

    def __initialize_camera(self):
        """Initialize the camera with configured settings from CameraConfig."""
        try:
            if not CameraHandler._picam2:
                CameraHandler._picam2 = Picamera2()

                # Configure camera
                camera_config = {
                    "size": (self.capture_width, self.capture_height),
                    "format": CameraConfig.CONTROLS['COLOR_FORMAT']
                }
                CameraHandler._picam2.configure(**camera_config)

                # Set camera controls from config
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
