from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter
import cv2
import numpy as np
from core.services.camera import CameraService

class CameraWidget(QWidget):
    """Widget for displaying camera feed"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_service = CameraService()
        self.pixmap = QPixmap()
        self.pixmap.fill(Qt.black)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

    def update_frame(self):
        """Update the displayed frame"""
        try:
            # Capture frame
            frame = self.camera_service.capture_frame()
            if frame is None:
                return

            # Convert color space and resize
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = self.camera_service.resize_frame(
                rgb_frame,
                self.width(),
                self.height()
            )

            # Create QImage and update pixmap
            image = QImage(
                resized_frame.data,
                self.width(),
                self.height(),
                QImage.Format_RGB888
            )
            self.pixmap = QPixmap.fromImage(image)
            self.update()

        except Exception as e:
            print(f"Frame update error: {e}")

    def paintEvent(self, event):
        """Paint the current frame"""
        if self.pixmap.isNull():
            return

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

    def capture_image(self, filename: str = "captured_image.jpg") -> bool:
        """Capture and save the current frame"""
        try:
            frame = self.camera_service.capture_frame()
            if frame is None:
                return False
            return self.camera_service.save_frame(frame, filename)
        except Exception as e:
            print(f"Error capturing image: {e}")
            return False

    def cleanup(self):
        """Cleanup widget resources"""
        self.timer.stop()
        self.camera_service.cleanup()
