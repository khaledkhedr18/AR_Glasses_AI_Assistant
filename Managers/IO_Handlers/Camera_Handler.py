from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt
import threading
import cv2
import numpy as np
import time
import os
from picamera2 import Picamera2
import libcamera

class CameraWidget(QWidget):
    def __init__(self, parent=None):
        """
        Initialize the CameraWidget.

        :param parent: Parent QWidget.
        """
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.picam2 = None
        self.pixmap = QPixmap()
        self.pixmap.fill(Qt.black)
        self.camera_lock = threading.Lock()
        self.capture_width = 1920
        self.capture_height = 1080
        self.frame_buffer = None
        self.last_frame_time = 0
        self.frame_rate = 30
        self.is_recording = False
        self.recording_thread = None
        self.initialize_camera()

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

                # Start frame update timer if used in a GUI context
                if self.parent():
                    self.timer = QTimer()
                    self.timer.timeout.connect(self.update_frame)
                    self.timer.start(1000 // self.frame_rate)  # ~33 FPS

        except Exception as e:
            print(f"Camera Initialization Error: {e}")
            self.pixmap = QPixmap(640, 480)
            self.pixmap.fill(Qt.red)

    def update_frame(self):
        """
        Capture a frame from the camera, convert it to RGB, resize it to the widget size,
        and update the widget with the new image.

        This method is called by a QTimer to continuously update the frame.
        """
        try:
            # Capture frame if enough time has passed since last capture
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0 / self.frame_rate:
                self.last_frame_time = current_time

                # Capture full resolution frame
                buffer = self.picam2.capture_array("main")
                self.frame_buffer = buffer  # Store the latest frame

                # Convert color space and resize for display
                rgb_buffer = cv2.cvtColor(buffer, cv2.COLOR_BGR2RGB)
                resized_buffer = cv2.resize(rgb_buffer, (self.width(), self.height()),
                                         interpolation=cv2.INTER_LANCZOS4)

                # Create QImage with correct dimensions
                from PyQt5.QtGui import QImage
                image = QImage(resized_buffer.data, self.width(), self.height(),
                             QImage.Format_RGB888)
                self.pixmap = QPixmap.fromImage(image)
                self.update()

        except Exception as e:
            print(f"Frame update error: {e}")

    def capture_frame(self):
        """
        Capture a single frame from the camera.

        Returns:
            numpy.ndarray: The captured frame as an numpy array, or None if capture failed
        """
        with self.camera_lock:
            try:
                return self.picam2.capture_array("main")
            except Exception as e:
                print(f"Error capturing frame: {e}")
                return None

    def capture_image(self, filename="captured_image.jpg", quality=90):
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
                cv2.imwrite(filename, buffer, [cv2.IMWRITE_JPEG_QUALITY, quality])
                print(f"Image saved as {filename}")
                return filename
            except Exception as e:
                print(f"Error capturing image: {e}")
                return None

    def capture_image_with_voice(self, audio_handler, model, timeout=30):
        """
        Captures an image when the user says 'take' using voice commands.

        Args:
            audio_handler: AudioHandler instance to process voice commands
            model: Speech recognition model
            timeout (int): Maximum wait time in seconds

        Returns:
            str or None: Path to the saved image if successful, None otherwise
        """
        try:
            print("Say 'take' to take a picture.")
            filename = f"capture_{int(time.time())}.jpg"
            max_attempts = 3
            attempts = 0

            start_time = time.time()
            while time.time() - start_time < timeout:
                audio_handler.speak("Please say 'take' to take a picture")
                print("Please say 'take' to take a picture")

                audio_command = audio_handler.get_audio(model)
                if audio_command and "take" in audio_command.strip().lower():
                    saved_path = self.capture_image(filename)
                    if saved_path and os.path.exists(saved_path):
                        return saved_path
                    else:
                        audio_handler.speak("Failed to capture image, please try again")
                        attempts += 1
                        print(f"Attempt No.: {attempts}")
                        if attempts >= max_attempts:
                            break
                elif audio_command and "cancel" in audio_command.strip().lower():
                    audio_handler.speak("Image capture canceled")
                    return None
                else:
                    attempts += 1
                    print(f"Attempt No.: {attempts}")
                    if attempts >= max_attempts:
                        break

            audio_handler.speak("Maximum attempts reached")
            return None

        except Exception as e:
            print(f"Camera error: {e}")
            return None

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
            print(f"Error setting camera parameters: {e}")
            return False

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
            print("Failed to load image for processing")
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
            print("Already recording")
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

            return True

        except Exception as e:
            print(f"Error starting recording: {e}")
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

        return True

    def paintEvent(self, event):
        """
        Reimplemented from QWidget.

        Paints the current pixmap onto the widget.

        Args:
            event: The paint event.
        """
        if self.pixmap.isNull():
            return

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

    def resizeEvent(self, event):
        """
        Reimplemented from QWidget.

        Handles widget resize events and updates the view.

        Args:
            event: The resize event.
        """
        self.update()
        super().resizeEvent(event)

    def cleanup(self):
        """
        Release camera resources.

        Should be called when the application is closing.
        """
        try:
            # Stop recording if in progress
            if self.is_recording:
                self.stop_recording()

            # Stop frame update timer if it exists
            if hasattr(self, 'timer'):
                self.timer.stop()

            # Close camera
            if self.picam2:
                self.picam2.close()
                self.picam2 = None

            print("Camera resources released")
        except Exception as e:
            print(f"Error during camera cleanup: {e}")
