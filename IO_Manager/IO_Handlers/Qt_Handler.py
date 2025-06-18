import time
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QSizePolicy)
from PyQt5.QtGui import QPixmap, QPainter, QImage
import cv2
from utils.config import OVERLAY_WIDGET_CONFIGS
from utils.logging import Logger
from utils.WorkerThread import WorkerThread

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
            self.io_manager = None  # Will be set by main application

            # Camera display properties
            self.camera_widget = None
            self.camera_pixmap = None
            self.frame_rate = 30
            self.camera_timer = None

            self.signals = CommunicationSignals()
            self.worker_threads = []
            self.is_fullscreen = False
            self.overlay_widgets = {}  # Store widget references by ID

            # Log initialization duration
            duration = time.time() - start_time
            self.logger.log_performance("qt_handler_init", duration)
            self.logger.info("Qt_Handler initialized successfully", duration=f"{duration:.3f}s")

        except Exception as e:
            self.logger.log_error_with_traceback("Qt_Handler initialization failed", e)
            raise

    def set_io_manager(self, io_manager):
        """Set the IO_Manager reference"""
        self.io_manager = io_manager
        self.logger.info("IO_Manager reference set in Qt_Handler")

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

            # Create camera widget (now directly in Qt_Handler instead of using QtCameraWidget)
            self.logger.debug("Initializing camera widget")
            self.camera_widget = QWidget()
            self.camera_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.camera_pixmap = QPixmap()
            self.camera_pixmap.fill(Qt.black)  # Initial black screen

            # Set up custom paint event for the camera widget
            class CameraDisplay(QWidget):
                def __init__(self, parent, pixmap_ref):
                    super().__init__(parent)
                    self.pixmap_ref = pixmap_ref
                    self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                def paintEvent(self, event):
                    if not self.pixmap_ref or self.pixmap_ref.isNull():
                        return

                    painter = QPainter(self)
                    painter.drawPixmap(0, 0, self.width(), self.height(), self.pixmap_ref)

                def resizeEvent(self, event):
                    self.update()
                    super().resizeEvent(event)

            self.camera_widget = CameraDisplay(None, self.camera_pixmap)
            main_layout.addWidget(self.camera_widget)

            # We remove the camera timer here - IO_Manager will control frame updates

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

    def update_camera_frame(self, frame):
        """
        Update the camera frame display with the provided frame.
        This method is called by IO_Manager with frames from Camera_Handler.

        Args:
            frame: OpenCV image frame to display
        """
        try:
            if frame is None:
                self.logger.warning("Received empty frame for display")
                return

            # Convert color space if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Assume BGR format from OpenCV
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame

            # Resize to widget dimensions
            height, width = rgb_frame.shape[:2]
            if self.camera_widget and self.camera_widget.width() > 0 and self.camera_widget.height() > 0:
                resized_frame = cv2.resize(
                    rgb_frame,
                    (self.camera_widget.width(), self.camera_widget.height()),
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
                self.camera_pixmap = QPixmap.fromImage(image)

                # Update the camera widget
                if self.camera_widget:
                    self.camera_widget.pixmap_ref = self.camera_pixmap
                    self.camera_widget.update()
        except Exception as e:
            self.logger.error(f"Error updating camera frame: {e}")

    def capture_image(self, filename=None):
        """
        Capture an image using IO_Manager

        Args:
            filename (str, optional): Path to save the image

        Returns:
            str: Path to saved image or None if failed
        """
        if not self.io_manager:
            self.logger.warning("Cannot capture image: IO_Manager not set")
            return None

        return self.io_manager.capture_image(filename)

    def create_overlay_widget(self, widget_type, config=None):
        """
        Create and configure a single overlay widget based on the specified type and configuration

        Args:
            widget_type (str): Type of widget to create ('status', 'user_speech', or 'ai_response')
            config (dict, optional): Custom configuration overrides

        Returns:
            QLabel: The created widget instance
        """
        self.logger.debug(f"Creating overlay widget: {widget_type}")
        start_time = time.time()

        if not self.main_window:
            self.logger.warning(f"Cannot create {widget_type} overlay: main_window is None")
            return None

        try:
            # Get default configuration for this widget type
            default_config = OVERLAY_WIDGET_CONFIGS.get(widget_type, {})

            # Merge with custom config if provided
            widget_config = default_config.copy()
            if config:
                widget_config.update(config)

            # Create label widget
            widget = QLabel(self.main_window)

            # Apply styling
            if "style" in widget_config:
                widget.setStyleSheet(widget_config["style"])

            # Set alignment
            if "alignment" in widget_config:
                alignment = widget_config["alignment"]
                if alignment == "center":
                    widget.setAlignment(Qt.AlignCenter)
                elif alignment == "left":
                    widget.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                elif alignment == "right":
                    widget.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # Set size
            if "size" in widget_config:
                widget.setFixedSize(*widget_config["size"])

            # Set position
            if "position" in widget_config:
                widget.move(*widget_config["position"])

            # Set word wrap
            if widget_config.get("word_wrap", False):
                widget.setWordWrap(True)

            # Set default text
            if "default_text" in widget_config:
                widget.setText(widget_config["default_text"])

            # Make label ignore mouse events
            widget.setAttribute(Qt.WA_TransparentForMouseEvents)

            # Raise widget to top
            widget.raise_()

            # Store reference to the widget based on its type
            if widget_type == "status":
                self.status_label = widget
            elif widget_type == "user_speech":
                self.user_speech_label = widget
            elif widget_type == "ai_response":
                self.ai_response_label = widget

            # Store reference in the overlay_widgets dictionary
            widget_id = f"{widget_type}_{id(widget)}"
            self.overlay_widgets[widget_id] = {
                "widget": widget,
                "type": widget_type,
                "config": widget_config
            }

            # Connect signals for standard widget types
            if widget_type == "ai_response" and hasattr(self, 'signals'):
                self.signals.update_ai_speech.connect(lambda text: self.update_ai_response(text))
            elif widget_type == "user_speech" and hasattr(self, 'signals'):
                self.signals.update_user_speech.connect(lambda text: self.update_user_speech(text))

            # Log performance
            duration = time.time() - start_time
            self.logger.log_performance(f"create_{widget_type}_overlay", duration)
            self.logger.debug(f"{widget_type} overlay created successfully", duration=f"{duration:.3f}s")

            return widget

        except Exception as e:
            self.logger.log_error_with_traceback(f"Error creating {widget_type} overlay widget", e)
            return None

    def create_overlay_widgets(self):
        """Create and configure the standard overlay elements"""
        self.logger.debug("Creating all standard overlay widgets")

        # Create the three standard widgets
        self.create_overlay_widget("status")
        self.create_overlay_widget("user_speech")
        self.create_overlay_widget("ai_response")

        # Start status check timer
        self.status_check_timer = QTimer(self.main_window)
        self.status_check_timer.timeout.connect(self.check_status)
        self.status_check_timer.start(1000)  # Check every second

        return {
            "status": self.status_label if hasattr(self, "status_label") else None,
            "user_speech": self.user_speech_label if hasattr(self, "user_speech_label") else None,
            "ai_response": self.ai_response_label if hasattr(self, "ai_response_label") else None
        }

    def check_status(self):
        """Default status check implementation - override if needed"""
        # This would typically call into a network service to check connectivity
        # For now, we'll just keep the status as is
        pass

    def update_status(self, status_text, is_online=None):
        """Alias for _update_network_status to maintain compatibility"""
        return self._update_network_status(status_text, is_online)

    def _update_network_status(self, status_text, is_online=None):
        """
        Update the status display

        Args:
            status_text (str): Status text to display
            is_online (bool, optional): If provided, adds ONLINE/OFFLINE indicator
        """
        if not hasattr(self, 'status_label'):
            self.logger.warning("Cannot update status: status_label not initialized")
            return False

        try:
            if is_online is not None:
                status = "ONLINE" if is_online else "OFFLINE"
                status_msg = f"{status_text} [{status}]"
                self.status_label.setText(status_msg)
                self.logger.debug(f"Updated status with connection state: {status_msg}")
            else:
                self.status_label.setText(status_text)
                self.logger.debug(f"Updated status: {status_text}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update status: {str(e)}")
            return False

    def update_user_speech(self, text):
        """Alias for _update_user_speech to maintain compatibility"""
        return self._update_user_speech(text)

    def _update_user_speech(self, text):
        """Update the user speech display with the given text"""
        self.logger.debug(f"Updating user speech: '{text}'")

        if not hasattr(self, 'user_speech_label'):
            self.logger.warning("Cannot update user speech: user_speech_label not initialized")
            return False

        try:
            self.user_speech_label.setText(f"You: {text}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update user speech: {str(e)}")
            return False

    def update_ai_response(self, text):
        """Alias for _update_ai_response to maintain compatibility"""
        return self._update_ai_response(text)

    def _update_ai_response(self, text):
        """Update the AI response display with the given text"""
        self.logger.debug(f"Updating AI response: '{text}'")

        if not hasattr(self, 'ai_response_label'):
            self.logger.warning("Cannot update AI response: ai_response_label not initialized")
            return False

        try:
            self.ai_response_label.setText(f"Assistant: {text}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update AI response: {str(e)}")
            return False

    def display_text_in_widget(self, text, widget_instance):
        """
        Display text in the specified widget instance

        Args:
            text (str): Text to display
            widget_instance (QLabel): Widget instance to update

        Returns:
            bool: True if successful, False otherwise
        """
        if not widget_instance:
            self.logger.warning("Cannot update widget: widget_instance is None")
            return False

        try:
            # Find widget in our dictionary to determine its type
            widget_type = None
            for widget_id, info in self.overlay_widgets.items():
                if info["widget"] == widget_instance:
                    widget_type = info["type"]
                    break

            # Use appropriate update method based on widget type
            if widget_type == "status":
                return self._update_network_status(text)
            elif widget_type == "user_speech":
                return self._update_user_speech(text)
            elif widget_type == "ai_response":
                return self._update_ai_response(text)
            else:
                # For any other widget, just set the text directly
                widget_instance.setText(text)
                return True

        except Exception as e:
            self.logger.error(f"Failed to update widget text: {str(e)}")
            return False

    def hide_widget(self, widget_instance):
        """Alias for hide_overlay_widget to maintain compatibility"""
        return self.hide_overlay_widget(widget_instance)

    def hide_overlay_widget(self, widget_instance):
        """
        Hide the specified widget

        Args:
            widget_instance: Widget instance to hide

        Returns:
            bool: True if successful, False otherwise
        """
        if not widget_instance:
            self.logger.warning("Cannot hide widget: widget_instance is None")
            return False

        try:
            widget_instance.hide()
            return True
        except Exception as e:
            self.logger.error(f"Failed to hide widget: {str(e)}")
            return False

    def hide_all_widgets(self):
        """Alias for hide_all_overlay_widgets to maintain compatibility"""
        return self.hide_all_overlay_widgets()

    def hide_all_overlay_widgets(self):
        """
        Hide all overlay widgets

        Returns:
            bool: True if successful, False otherwise
        """
        success = True
        for widget_id, info in self.overlay_widgets.items():
            try:
                info["widget"].hide()
            except Exception as e:
                self.logger.error(f"Failed to hide widget {widget_id}: {str(e)}")
                success = False

        return success

    def show_widget(self, widget_instance):
        """
        Show the specified widget

        Args:
            widget_instance: Widget instance to show

        Returns:
            bool: True if successful, False otherwise
        """
        if not widget_instance:
            self.logger.warning("Cannot show widget: widget_instance is None")
            return False

        try:
            widget_instance.show()
            widget_instance.raise_()
            return True
        except Exception as e:
            self.logger.error(f"Failed to show widget: {str(e)}")
            return False

    def display_image_in_widget(self, image_path, widget_instance):
        """
        Display an image in the specified widget

        Args:
            image_path (str): Path to the image file
            widget_instance (QLabel): Widget instance to update

        Returns:
            bool: True if successful, False otherwise
        """
        if not widget_instance:
            self.logger.warning("Cannot display image: widget_instance is None")
            return False

        try:
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                self.logger.error(f"Failed to load image from {image_path}")
                return False

            # Resize pixmap to fit the label while maintaining aspect ratio
            pixmap = pixmap.scaled(
                widget_instance.width(),
                widget_instance.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            # Set pixmap to label
            widget_instance.setPixmap(pixmap)
            return True
        except Exception as e:
            self.logger.error(f"Failed to display image: {str(e)}")
            return False

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
            # Stop camera update timer
            if self.camera_timer and self.camera_timer.isActive():
                self.logger.debug("Stopping camera update timer")
                self.camera_timer.stop()

            # Stop all worker threads
            worker_count = len(self.worker_threads)
            self.logger.debug(f"Stopping {worker_count} worker threads")
            for worker in self.worker_threads[:]:
                try:
                    worker.stop()
                    worker.wait(1000)  # Wait up to 1 second
                except Exception as e:
                    self.logger.warning(f"Error stopping worker thread: {str(e)}")

            # Stop status check timer
            if hasattr(self, 'status_check_timer'):
                self.logger.debug("Stopping status check timer")
                self.status_check_timer.stop()

            duration = time.time() - start_time
            self.logger.log_performance("qt_handler_cleanup", duration)
            self.logger.info("Qt_Handler cleanup completed", duration=f"{duration:.3f}s")

        except Exception as e:
            self.logger.log_error_with_traceback("Error during Qt_Handler cleanup", e)

    # Add this new method to the Qt_Handler class
    def delete_overlay_widget(self, widget_instance):
        """
        Permanently delete an overlay widget from the application.

        Args:
            widget_instance: The widget to delete

        Returns:
            bool: True if successfully deleted, False otherwise
        """
        if not widget_instance:
            self.logger.warning("Cannot delete widget: widget_instance is None")
            return False

        try:
            # Find the widget in our dictionary
            widget_id_to_delete = None
            widget_type = None

            for widget_id, info in self.overlay_widgets.items():
                if info["widget"] == widget_instance:
                    widget_id_to_delete = widget_id
                    widget_type = info["type"]
                    break

            if not widget_id_to_delete:
                self.logger.warning("Widget not found in overlay_widgets dictionary")
                return False

            # Disconnect any signals associated with the widget
            if widget_type == "ai_response" and hasattr(self, 'signals'):
                try:
                    self.signals.update_ai_speech.disconnect()
                except TypeError:
                    pass  # Signal might not be connected
            elif widget_type == "user_speech" and hasattr(self, 'signals'):
                try:
                    self.signals.update_user_speech.disconnect()
                except TypeError:
                    pass  # Signal might not be connected

            # Remove specific widget references
            if widget_type == "status" and hasattr(self, 'status_label') and self.status_label == widget_instance:
                self.status_label = None
            elif widget_type == "user_speech" and hasattr(self, 'user_speech_label') and self.user_speech_label == widget_instance:
                self.user_speech_label = None
            elif widget_type == "ai_response" and hasattr(self, 'ai_response_label') and self.ai_response_label == widget_instance:
                self.ai_response_label = None

            # Hide the widget before deletion
            widget_instance.hide()

            # Remove from main dictionary
            del self.overlay_widgets[widget_id_to_delete]

            # Schedule the widget for deletion
            widget_instance.deleteLater()

            self.logger.info(f"Successfully deleted {widget_type} widget")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete widget: {str(e)}")
            return False
