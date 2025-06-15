from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QPixmap, QPainter, QImage
import cv2
from IO_Manager.IO_Handlers.Camera_Handler import CameraHandler
from utils.logging import Logger

class QtCameraWidget(QWidget):
    """
    Qt-specific widget for displaying camera feed using the framework-agnostic CameraHandler.
    Acts as a bridge between pure Python CameraHandler and Qt GUI.
    """

    def __init__(self, parent=None):
        """Initialize the QtCameraWidget."""
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create the core camera handler
        self.camera_handler = CameraHandler()

        # Create Qt-specific components
        self.pixmap = QPixmap()
        self.pixmap.fill(Qt.black)
        self.logger = Logger()

        # Set up frame update timer
        self.frame_rate = 30
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // self.frame_rate)

    def update_frame(self):
        """Update the displayed frame from the camera."""
        try:
            # Capture frame from camera handler
            frame = self.camera_handler.capture_frame()
            if frame is not None:
                # Convert color space if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Assume BGR format from OpenCV
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame

                # Resize to widget dimensions
                resized_frame = cv2.resize(
                    rgb_frame,
                    (self.width(), self.height()),
                    interpolation=cv2.INTER_LANCZOS4
                )

                # Create QImage and QPixmap
                height, width, channel = resized_frame.shape
                bytes_per_line = 3 * width
                image = QImage(
                    resized_frame.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_RGB888
                )
                self.pixmap = QPixmap.fromImage(image)
                self.update()

        except Exception as e:
            self.logger.error(f"Error updating camera frame: {e}")

    def paintEvent(self, event):
        """Paint the camera frame on widget."""
        if self.pixmap.isNull():
            return

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

    def resizeEvent(self, event):
        """Handle widget resize events."""
        self.update()
        super().resizeEvent(event)

    def capture_image(self, filename=None):
        """Proxy to the camera handler's capture_image method."""
        return self.camera_handler.capture_image(filename)

    def capture_frame(self):
        """Proxy to the camera handler's capture_frame method."""
        return self.camera_handler.capture_frame()

    def process_image(self, image, processing_level="medium"):
        """Proxy to the camera handler's process_image method."""
        return self.camera_handler.process_image(image, processing_level)

    def cleanup(self):
        """Clean up resources."""
        if self.timer.isActive():
            self.timer.stop()

        if self.camera_handler:
            self.camera_handler.cleanup()
