import threading
import cv2
from picamera2 import Picamera2
import libcamera
from Utils.Config import CAMERA_CONFIG
from Utils.Logging import Logger
import os


class CameraHandler:
    def __init__(self):
        """
        Initialize the Camera
        """
        self.logger = Logger()
        self.camera_lock = threading.Lock()
        self.frame_buffer = None
        self.picam2 = None
        self.capture_width = CAMERA_CONFIG['CAPTURE_WIDTH']
        self.capture_height = CAMERA_CONFIG['CAPTURE_HEIGHT']
        self.quality = CAMERA_CONFIG['IMAGE_QUALITY']
        self.save_dir = CAMERA_CONFIG['SAVE_DIRECTORY']
        self.saved_image_name = CAMERA_CONFIG['IMAGE_FILENAME']
        self.image_save_path = None
        self.__initialize_camera()

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

    def __initialize_camera(self):
        """
        Initialize the camera with native sensor resolution and proper color format.

        This method configures the camera with the native sensor resolution and
        RGB888 color format. It also sets the autofocus mode to continuous and
        starts the camera. The capture dimensions are extracted from the
        configuration and stored as instance variables.

        If there is an error during initialization, the `pixmap` is filled with
        red color to indicate the error.
        """
        try:
            if self.picam2 is None:
                self.picam2 = Picamera2()
                config = self.picam2.create_preview_configuration(
                    main={
                        "size": (self.capture_width, self.capture_height),
                        "format": CAMERA_CONFIG['COLOR_FORMAT']
                    },
                    controls={
                        "AwbEnable": CAMERA_CONFIG['AWB_ENABLE'],
                        "AeEnable": CAMERA_CONFIG['AE_ENABLE'],
                        "ExposureTime": CAMERA_CONFIG['DEFAULT_EXPOSURE'],
                        "AnalogueGain": CAMERA_CONFIG['DEFAULT_GAIN'],
                        "FrameDurationLimits": (CAMERA_CONFIG['FRAME_DURATION'],
                                                CAMERA_CONFIG['FRAME_DURATION'])
                    },
                    transform=libcamera.Transform(
                        hflip=CAMERA_CONFIG['HFLIP'],
                        vflip=CAMERA_CONFIG['VFLIP']
                    )
                )
                self.picam2.configure(config)

                # Set default focus mode
                mode = CAMERA_CONFIG['DEFAULT_FOCUS_MODE']
                if mode == 'continuous':
                    focus_mode = libcamera.controls.AfModeEnum.Continuous
                elif mode == 'auto':
                    focus_mode = libcamera.controls.AfModeEnum.Auto
                else:
                    focus_mode = libcamera.controls.AfModeEnum.Manual

                self.picam2.set_controls({"AfMode": focus_mode})
                self.picam2.start()

                self.capture_width = config["main"]["size"][0]
                self.capture_height = config["main"]["size"][1]
                self.logger.info("Camera initialized successfully")

        except Exception as e:
            self.logger.log_error_with_traceback("Camera Initialization Error", e)
