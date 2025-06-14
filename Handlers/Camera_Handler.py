import threading
import cv2
from picamera2 import Picamera2
import libcamera
import numpy as np


class CameraWidget:
    def __init__(self, parent=None):
        """
        Initialize the CameraWidget.

        :param parent: Parent QWidget.
        """

        self.picam2 = None
        self.camera_lock = threading.Lock()
        self.capture_width = 1920
        self.capture_height = 1080
        self.quality = 90
        #self.frame_rate = 30
        self.image_name = f"Saved_Images/captured_image.jpg"
        self.initialize_camera()

    def capture_image(self):
        """
        Capture an image from the camera and save it to a file.

        Args:
            filename (str): The filename to save the image to
            quality (int): JPEG quality setting (0-100)

        Returns:
            str or None: The filename if the capture is successful, or None otherwise
        """
        with self.camera_lock:
            try:
                # Capture full resolution image
                buffer = self.picam2.capture_array("main")
                cv2.imwrite(self.image_name, buffer, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                print(f"Image saved as {self.image_name}")
                return self.image_name
            except Exception as e:
                print(f"Error capturing image: {e}")
                return None

    def initialize_camera(self):
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
                # Use sensor resolution and proper color format
                config = self.picam2.create_preview_configuration(
                    main={
                        "size": (self.capture_width, self.capture_height),  # Native sensor mode for better FPS
                        "format": "RGB888"},
                    controls={
                        "AwbEnable": True,
                        "AeEnable": True,
                        "ExposureTime": 10000,  # Adjust based on lighting
                        "AnalogueGain": 1.0,
                        "FrameDurationLimits": (33333, 33333)  # 30fps
                    },
                    transform=libcamera.Transform(hflip=1, vflip=1)  # Adjust based on mounting
                )
                self.picam2.configure(config)

                # Set autofocus controls
                controls = {"AfMode": libcamera.controls.AfModeEnum.Continuous}
                self.picam2.set_controls(controls)
                self.picam2.start()

                # Get actual capture dimensions
                self.capture_width = config["main"]["size"][0]
                self.capture_height = config["main"]["size"][1]

            """
                # Start frame update timer if used in a GUI context
                if self.parent():
                    self.timer = QTimer()
                    self.timer.timeout.connect(self.update_frame)
                    self.timer.start(1000 // self.frame_rate)  # ~33 FPS
            """

        except Exception as e:
            print(f"Camera Initialization Error: {e}")