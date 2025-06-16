import sys
import cv2
import numpy as np
import threading
import time
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QObject, QEventLoop
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, QSizePolicy,
                            QPushButton, QVBoxLayout, QHBoxLayout, QFrame)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QColor

from IO_Manager.IO_Handlers.Camera_Handler import CameraWidget
from Utils.logging import Logger

# Check if on Raspberry Pi or development machine
try:
    from picamera2 import Picamera2
    import libcamera
    ON_PI = True
except ImportError:
    ON_PI = False

class CommunicationSignals(QObject):
    """Signals for communication between UI components"""
    update_output = pyqtSignal(str)
    update_ai_speech = pyqtSignal(str)
    update_user_speech = pyqtSignal(str)
    processing_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

class WorkerThread(QThread):
    """Thread for background tasks with result signaling capability"""
    finished = pyqtSignal()
    result = pyqtSignal(object)

    def __init__(self, task_func, *args, task_name=None):
        """
        Initialize a WorkerThread with a task function and arguments.

        Args:
            task_func: The function to run in the separate thread
            *args: The arguments to pass to the function
            task_name: Optional name for logging purposes
        """
        super().__init__()
        self.task_func = task_func
        self.args = args
        self._is_running = True
        self.task_name = task_name or task_func.__name__
        self.logger = Logger()

    def run(self):
        """Run the task function in a separate thread."""
        if self._is_running:
            start_time = time.time()
            try:
                self.logger.info(f"Starting worker task: {self.task_name}")
                result = self.task_func(*self.args)
                self.result.emit(result)

                # Log performance data
                duration = time.time() - start_time
                self.logger.log_performance(f"worker_{self.task_name}", duration)
                self.logger.info(f"Completed worker task: {self.task_name}", duration=f"{duration:.3f}s")
            except Exception as e:
                self.logger.log_error_with_traceback(f"Error in worker task: {self.task_name}", e)
            finally:
                self.finished.emit()

    def stop(self):
        """Stop the thread safely."""
        self.logger.debug(f"Stopping worker thread: {self.task_name}")
        self._is_running = False
        self.quit()
        self.wait()


class AudioVisualizer(QWidget):
    """Widget for visualizing audio levels"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(30)
        self.level = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.decrease_level)
        self.timer.start(50)  # Update every 50ms
        self.logger = Logger()
        self.logger.debug("AudioVisualizer initialized")

    def set_level(self, level):
        """Set the current audio level (0-100)"""
        self.level = min(max(level, 0), 100)
        self.update()

    def decrease_level(self):
        """Gradually decrease the level for visual effect"""
        if self.level > 0:
            self.level = max(0, self.level - 3)
            self.update()

    def paintEvent(self, event):
        """Draw the audio level visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        # Draw level bar
        if self.level > 0:
            bar_width = int(self.width() * (self.level / 100))

            # Choose color based on level
            if self.level < 30:
                color = QColor(0, 200, 0)  # Green for low levels
            elif self.level < 70:
                color = QColor(200, 200, 0)  # Yellow for medium levels
            else:
                color = QColor(200, 0, 0)  # Red for high levels

            painter.fillRect(0, 0, bar_width, self.height(), color)

class Qt_Handler:
    """
    Handles all Qt GUI-related functionality.
    This class provides a unified interface for creating and managing
    the Qt-based user interface components of the application.
    """

    def __init__(self):
        """Initialize the Qt GUI handler with application instance"""
        # Initialize logger
        self.logger = Logger()
        self.logger.info("Initializing Qt_Handler")

        start_time = time.time()

        try:
            # Create Qt application instance if not already created
            self.app = QApplication([]) if not QApplication.instance() else QApplication.instance()
            self.main_window = None
            self.camera_widget = None
            self.audio_visualizer = None
            self.signals = CommunicationSignals()
            self.worker_threads = []
            self.is_fullscreen = False

            # Log initialization duration
            duration = time.time() - start_time
            self.logger.log_performance("qt_handler_init", duration)
            self.logger.info("Qt_Handler initialized successfully", duration=f"{duration:.3f}s")

        except Exception as e:
            self.logger.log_error_with_traceback("Qt_Handler initialization failed", e)
            raise

    def create_window(self, title="AR Glasses Assistant", fullscreen=True):
        """
        Create the main application window with default layout

        Args:
            title (str): Window title
            fullscreen (bool): Whether to show in fullscreen mode

        Returns:
            QMainWindow: The created main window
        """
        self.logger.info(f"Creating main window: title='{title}', fullscreen={fullscreen}")
        start_time = time.time()

        try:
            self.main_window = QMainWindow()
            self.main_window.setWindowTitle(title)

            # Create central widget and layout
            central_widget = QWidget()
            self.main_window.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)

            # Create camera widget using the imported CameraWidget class
            self.logger.debug("Initializing camera widget")
            self.camera_widget = CameraWidget()
            main_layout.addWidget(self.camera_widget)

            # Add standard overlay widgets
            self.create_overlay_widgets()

            # Set window properties
            self.is_fullscreen = fullscreen
            if fullscreen:
                self.main_window.showFullScreen()
                self.logger.debug("Window set to fullscreen mode")
            else:
                self.main_window.setGeometry(100, 100, 800, 600)
                self.logger.debug("Window set to windowed mode (800x600)")

            # Log window creation duration
            duration = time.time() - start_time
            self.logger.log_performance("window_creation", duration)
            self.logger.info("Main window created successfully", duration=f"{duration:.3f}s")

            return self.main_window

        except Exception as e:
            self.logger.log_error_with_traceback("Error creating main window", e)
            return None

    def create_overlay_widgets(self):
        """Create and configure the standard overlay elements"""
        self.logger.debug("Creating overlay widgets")
        start_time = time.time()

        if not self.main_window:
            self.logger.warning("Cannot create overlays: main_window is None")
            return

        try:
            # Status indicator
            self.status_label = QLabel(self.main_window)
            self.status_label.setStyleSheet("""
                QLabel {
                    color: white;
                    background-color: rgba(0,0,0,200);
                    border-radius: 5px;
                    padding: 5px;
                    font: bold 14px;
                }
            """)
            self.status_label.setAlignment(Qt.AlignCenter)
            self.status_label.setFixedSize(200, 40)
            self.status_label.move(300, 20)
            self.status_label.setText("Status: INITIALIZING")
            self.status_label.raise_()

            # User speech display
            self.user_speech_label = QLabel(self.main_window)
            self.user_speech_label.setStyleSheet("""
                QLabel {
                    color: white;
                    background-color: rgba(0,0,0,200);
                    border-radius: 5px;
                    padding: 8px;
                    font: 12px;
                }
            """)
            self.user_speech_label.setFixedSize(300, 100)
            self.user_speech_label.move(480, 360)
            self.user_speech_label.setWordWrap(True)
            self.user_speech_label.setText("You: ")
            self.user_speech_label.raise_()

            # AI response display
            self.ai_response_label = QLabel(self.main_window)
            self.ai_response_label.setStyleSheet("""
                QLabel {
                    color: white;
                    background-color: rgba(0,0,0,200);
                    border-radius: 5px;
                    padding: 8px;
                    font: 12px;
                }
            """)
            self.ai_response_label.setFixedSize(300, 100)
            self.ai_response_label.move(20, 360)
            self.ai_response_label.setWordWrap(True)
            self.ai_response_label.setText("Assistant: Ready")
            self.ai_response_label.raise_()

            # Audio visualizer
            self.audio_visualizer = AudioVisualizer(self.main_window)
            self.audio_visualizer.setGeometry(20, 480, 760, 30)
            self.audio_visualizer.raise_()

            # Connect signals
            self.signals.update_ai_speech.connect(self.update_ai_response)
            self.signals.update_user_speech.connect(self.update_user_speech)

            # Make labels ignore mouse events
            for label in [self.status_label, self.user_speech_label, self.ai_response_label]:
                label.setAttribute(Qt.WA_TransparentForMouseEvents)

            # Start status check timer
            self.status_check_timer = QTimer(self.main_window)
            self.status_check_timer.timeout.connect(self.check_status)
            self.status_check_timer.start(1000)  # Check every second

            # Log performance
            duration = time.time() - start_time
            self.logger.log_performance("create_overlays", duration)
            self.logger.debug("Overlay widgets created successfully", duration=f"{duration:.3f}s")

        except Exception as e:
            self.logger.log_error_with_traceback("Error creating overlay widgets", e)

    def update_status(self, status_text, is_online=None):
        """
        Update the status display

        Args:
            status_text (str): Status text to display
            is_online (bool, optional): If provided, adds ONLINE/OFFLINE indicator
        """
        if not hasattr(self, 'status_label'):
            self.logger.warning("Cannot update status: status_label not initialized")
            return

        try:
            if is_online is not None:
                status = "ONLINE" if is_online else "OFFLINE"
                status_msg = f"{status_text} [{status}]"
                self.status_label.setText(status_msg)
                self.logger.debug(f"Updated status with connection state: {status_msg}")
            else:
                self.status_label.setText(status_text)
                self.logger.debug(f"Updated status: {status_text}")
        except Exception as e:
            self.logger.error(f"Failed to update status: {str(e)}")

    def check_status(self):
        """Default status check implementation - override if needed"""
        # This would typically call into a network service to check connectivity
        # For now, we'll just keep the status as is
        pass

    def update_user_speech(self, text):
        """Update the user speech display with the given text"""
        self.logger.debug(f"Updating user speech: '{text}'")

        if not hasattr(self, 'user_speech_label'):
            self.logger.warning("Cannot update user speech: user_speech_label not initialized")
            return

        try:
            self.user_speech_label.setText(f"You: {text}")

            # Simulate audio level based on text length
            if hasattr(self, 'audio_visualizer') and self.audio_visualizer:
                # Generate a level proportional to the string length (for demo)
                level = min(len(text) * 5, 100)
                self.audio_visualizer.set_level(level)
        except Exception as e:
            self.logger.error(f"Failed to update user speech: {str(e)}")

    def update_ai_response(self, text):
        """Update the AI response display with the given text"""
        self.logger.debug(f"Updating AI response: '{text}'")

        if not hasattr(self, 'ai_response_label'):
            self.logger.warning("Cannot update AI response: ai_response_label not initialized")
            return

        try:
            self.ai_response_label.setText(f"Assistant: {text}")
        except Exception as e:
            self.logger.error(f"Failed to update AI response: {str(e)}")

    def create_worker(self, task_func, *args, task_name=None):
        """
        Create a worker thread for background tasks

        Args:
            task_func: Function to run
            *args: Arguments to pass to the function
            task_name: Optional task name for logging

        Returns:
            WorkerThread: The created worker thread
        """
        name = task_name or task_func.__name__
        self.logger.debug(f"Creating worker thread for task: {name}")

        try:
            worker = WorkerThread(task_func, *args, task_name=name)
            worker.finished.connect(lambda: self.cleanup_worker(worker))
            self.worker_threads.append(worker)
            return worker
        except Exception as e:
            self.logger.error(f"Failed to create worker thread: {str(e)}")
            return None

    def cleanup_worker(self, worker):
        """Remove a worker thread from the tracking list when it finishes"""
        if worker in self.worker_threads:
            task_name = getattr(worker, 'task_name', 'unknown')
            self.logger.debug(f"Cleaning up worker thread: {task_name}")
            self.worker_threads.remove(worker)

    def get_camera_widget(self):
        """Get the camera widget instance"""
        return self.camera_widget

    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode"""
        if not self.main_window:
            self.logger.warning("Cannot toggle fullscreen: main_window is None")
            return

        try:
            if self.is_fullscreen:
                self.main_window.showNormal()
                self.logger.info("Switched to windowed mode")
            else:
                self.main_window.showFullScreen()
                self.logger.info("Switched to fullscreen mode")

            self.is_fullscreen = not self.is_fullscreen

        except Exception as e:
            self.logger.error(f"Failed to toggle fullscreen mode: {str(e)}")

    def adjust_overlay_positions(self):
        """Adjust overlay positions when window size changes"""
        if not self.main_window:
            self.logger.warning("Cannot adjust overlay positions: main_window is None")
            return

        try:
            start_time = time.time()
            width = self.main_window.width()
            height = self.main_window.height()

            self.logger.debug(f"Adjusting overlays for window size: {width}x{height}")

            # Status in top center
            if hasattr(self, 'status_label'):
                self.status_label.move((width - self.status_label.width()) // 2, 20)

            # User speech in bottom right
            if hasattr(self, 'user_speech_label'):
                self.user_speech_label.move(
                    width - self.user_speech_label.width() - 20,
                    height - self.user_speech_label.height() - 20
                )

            # AI response in bottom left
            if hasattr(self, 'ai_response_label'):
                self.ai_response_label.move(20, height - self.ai_response_label.height() - 20)

            # Audio visualizer at bottom
            if hasattr(self, 'audio_visualizer') and self.audio_visualizer:
                self.audio_visualizer.setGeometry(
                    20, height - 50, width - 40, 30
                )

            # Log performance for UI adjustment
            duration = time.time() - start_time
            self.logger.log_performance("adjust_overlays", duration)

        except Exception as e:
            self.logger.error(f"Failed to adjust overlay positions: {str(e)}")

    def run(self):
        """Run the application main loop"""
        self.logger.info("Starting Qt application main loop")

        if not self.main_window:
            self.logger.error("Cannot run application: main_window is None")
            return 1

        try:
            self.main_window.show()
            self.logger.info("Qt application event loop started")
            return_code = self.app.exec_()
            self.logger.info(f"Qt application event loop ended with code: {return_code}")
            return return_code
        except Exception as e:
            self.logger.log_error_with_traceback("Error in Qt application main loop", e)
            return 1

    def cleanup(self):
        """
        Clean up resources used by the handler

        This should be called before application exit
        """
        self.logger.info("Starting Qt_Handler cleanup")
        start_time = time.time()

        try:
            # Stop all worker threads
            worker_count = len(self.worker_threads)
            self.logger.debug(f"Stopping {worker_count} worker threads")
            for worker in self.worker_threads[:]:
                try:
                    worker.stop()
                    worker.wait(1000)  # Wait up to 1 second
                except Exception as e:
                    self.logger.warning(f"Error stopping worker thread: {str(e)}")

            # Clean up camera if it exists
            if self.camera_widget:
                self.logger.debug("Cleaning up camera widget")
                self.camera_widget.cleanup()

            # Stop timers
            if hasattr(self, 'status_check_timer'):
                self.logger.debug("Stopping status check timer")
                self.status_check_timer.stop()

            duration = time.time() - start_time
            self.logger.log_performance("qt_handler_cleanup", duration)
            self.logger.info("Qt_Handler cleanup completed", duration=f"{duration:.3f}s")

        except Exception as e:
            self.logger.log_error_with_traceback("Error during Qt_Handler cleanup", e)
