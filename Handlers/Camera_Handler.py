import threading
import cv2
from picamera2 import Picamera2
import libcamera
import os

class CameraWidget:
    def __init__(self):
        """
        Initialize the CameraWidget.

        :param parent: Parent QWidget.
        """
        self.picam2 = None
        self.camera_lock = threading.Lock()
        self.capture_width = 1920
        self.capture_height = 1080
        self.quality = 90
        self.save_dir = "Saved_Images"
        self.image_save_path = None
        self.__initialize_camera()

    def capture_image(self):
        """
        Capture an image from the camera and save it to a static path.
        Returns: str or None: The saved image path if successful, None if failed
        """

        # Ensure the save directory exists
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Use os.path.join for cross-platform compatibility
        self.image_save_path = os.path.join(self.save_dir, "captured_image.jpg")

        with self.camera_lock:
            try:
                # Capture full resolution image
                buffer = self.picam2.capture_array("main")
                # Save the image with specified quality
                cv2.imwrite(self.image_save_path, buffer, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                print(f"Image saved as {self.image_save_path}")
                return self.image_save_path
            except Exception as e:
                print(f"Error capturing image: {e}")
                return None

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


        except Exception as e:
            print(f"Camera Initialization Error: {e}")