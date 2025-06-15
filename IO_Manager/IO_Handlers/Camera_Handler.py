import threading
import cv2
import numpy as np
import time
import os
from utils.logging import Logger

# Conditionally import PiCamera or fallback to OpenCV
try:
    from picamera2 import Picamera2
    import libcamera
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class CameraHandler:
    """
    Framework-agnostic camera handling class for image capture and processing.
    """

    def __init__(self):
        """Initialize the Camera Handler."""
        self.logger = Logger()
        self.logger.info("Initializing Camera Handler")

        self.camera_lock = threading.Lock()
        self.capture_width = 1920
        self.capture_height = 1080
        self.frame_buffer = None
        self.last_frame_time = 0
        self.frame_rate = 30
        self.is_recording = False
        self.recording_thread = None
        self.picam2 = None

        # Initialize camera if available
        if PICAMERA_AVAILABLE:
            self.initialize_camera()

    def initialize_camera(self):
        """Initialize the camera with appropriate settings."""
        try:
            if self.picam2 is None and PICAMERA_AVAILABLE:
                self.picam2 = Picamera2()

                # Use sensor resolution and proper color format
                config = self.picam2.create_preview_configuration(
                    main={
                        "size": (self.capture_width, self.capture_height),
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

                # Set autofocus controls
                controls = {"AfMode": libcamera.controls.AfModeEnum.Continuous}
                self.picam2.set_controls(controls)
                self.picam2.start()

                # Get actual capture dimensions
                self.capture_width = config["main"]["size"][0]
                self.capture_height = config["main"]["size"][1]

                self.logger.info(f"Camera initialized: {self.capture_width}x{self.capture_height}")
                return True

        except Exception as e:
            self.logger.error(f"Camera initialization error: {str(e)}")
            return False

        return False

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
                self.logger.error(f"Error capturing frame: {str(e)}")
                return None

    def capture_image(self, filename=None):
        """
        Capture an image and save it to a file.

        Args:
            filename (str, optional): The filename to save the image to

        Returns:
            str or None: The filename if the capture is successful, or None otherwise
        """
        with self.camera_lock:
            try:
                if not self.picam2:
                    self.logger.warning("Camera not initialized")
                    return None

                # Generate filename if not provided
                if not filename:
                    os.makedirs("Saved_Images", exist_ok=True)
                    filename = f"Saved_Images/capture_{int(time.time())}.jpg"

                # Capture full resolution image
                buffer = self.picam2.capture_array("main")
                cv2.imwrite(filename, buffer, [cv2.IMWRITE_JPEG_QUALITY, 90])
                self.logger.info(f"Image saved: {filename}")

                return filename

            except Exception as e:
                self.logger.error(f"Error capturing image: {str(e)}")
                return None

    def process_image(self, image, processing_level="medium"):
        """
        Process an image for better text recognition.

        Args:
            image: OpenCV image or path to image
            processing_level (str): "low", "medium", or "high"

        Returns:
            numpy.ndarray: Processed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image

        if img is None:
            self.logger.error("Failed to load image for processing")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if processing_level == "low":
            return gray

        # Medium processing (default)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if processing_level == "medium":
            return thresh

        # High processing
        if processing_level == "high":
            processed = cv2.dilate(thresh, np.ones((1, 1), np.uint8), iterations=1)
            processed = cv2.erode(processed, np.ones((1, 1), np.uint8), iterations=1)

            # Edge enhancement
            edges = cv2.Canny(processed, 50, 150)
            processed = cv2.addWeighted(processed, 0.8, edges, 0.2, 0)

            return processed

        # Default fallback
        return thresh

    def set_camera_parameters(self, exposure=None, gain=None, focus_mode=None):
        """
        Set camera parameters like exposure, gain and focus mode.

        Args:
            exposure (int, optional): Exposure time in microseconds
            gain (float, optional): Analog gain value
            focus_mode (str, optional): Focus mode (auto, continuous, manual)

        Returns:
            bool: True if parameters were set successfully, False otherwise
        """
        if not self.picam2:
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
            return True

        except Exception as e:
            self.logger.error(f"Error setting camera parameters: {str(e)}")
            return False

    def start_recording(self, output_file="video.mp4", fps=30, duration=None):
        """
        Start recording video from the camera.

        Args:
            output_file (str): Path to save the video
            fps (int): Frames per second
            duration (float, optional): Recording duration in seconds

        Returns:
            bool: True if recording started successfully, False otherwise
        """
        if self.is_recording:
            self.logger.warning("Already recording")
            return False

        try:
            # Get camera resolution
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_file,
                fourcc,
                fps,
                (self.capture_width, self.capture_height)
            )

            self.is_recording = True
            self.recording_stop_event = threading.Event()

            def record_thread_func():
                start_time = time.time()
                while self.is_recording and not self.recording_stop_event.is_set():
                    if duration and time.time() - start_time > duration:
                        break

                    frame = self.capture_frame()
                    if frame is not None:
                        self.video_writer.write(frame)
                    time.sleep(1/fps)

                # Clean up
                self.video_writer.release()
                self.is_recording = False

            # Start recording in a separate thread
            self.recording_thread = threading.Thread(target=record_thread_func)
            self.recording_thread.daemon = True
            self.recording_thread.start()

            self.logger.info(f"Started recording to {output_file}")
            return True

        except Exception as e:
            self.logger.error(f"Error starting recording: {str(e)}")
            self.is_recording = False
            return False

    def stop_recording(self):
        """
        Stop current video recording.

        Returns:
            bool: True if recording was stopped, False if no recording in progress
        """
        if not self.is_recording:
            return False

        self.recording_stop_event.set()
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)

        self.logger.info("Recording stopped")
        return True

    def cleanup(self):
        """Release camera resources."""
        try:
            # Stop recording if in progress
            if self.is_recording:
                self.stop_recording()

            # Close camera
            if self.picam2:
                self.picam2.close()
                self.picam2 = None

            self.logger.info("Camera resources released")
            return True
        except Exception as e:
            self.logger.error(f"Error during camera cleanup: {str(e)}")
            return False
