import time
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, QVBoxLayout, QSizePolicy)
from PyQt5.QtGui import QPixmap, QPainter, QImage
import cv2
from utils.Config import OVERLAY_WIDGET_CONFIGS
from utils.Logging import Logger
from utils.WorkerThread import create_worker


# Check if on Raspberry Pi or development machine
try:
    from picamera2 import Picamera2
    import libcamera
    ON_PI = True
except ImportError:
    ON_PI = False

class CommunicationSignals(QObject):
    """
    Dynamic signal system for widget communication

    This class allows for creating custom signals at runtime
    """
    # Default signal that can be used to update any widget
    update_widget = pyqtSignal(str, object)  # text, widget_reference

    def __init__(self):
        super().__init__()
        self._custom_signals = {}

    def create_signal(self, signal_name):
        """
        Create a custom signal dynamically

        Args:
            signal_name (str): Name for the new signal

        Returns:
            pyqtSignal: The created signal
        """
        if hasattr(self, signal_name):
            return getattr(self, signal_name)

        # Create a new signal
        new_signal = pyqtSignal(str)
        setattr(self, signal_name, new_signal)
        self._custom_signals[signal_name] = new_signal
        return new_signal

    def get_signal(self, signal_name):
        """Get an existing signal by name"""
        if hasattr(self, signal_name):
            return getattr(self, signal_name)
        return None

class QtHandler:
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


    def create_overlay_widget(self, widget_type, config=None):
        """Create and configure an overlay widget based on the specified type and configuration"""
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

            # Calculate position based on special values
            position = widget_config.get("position")
            if position == "center_top":
                # Center horizontally, near top
                x = (self.main_window.width() - widget.width()) // 2
                y = 20
                widget.move(x, y)
            elif position == "bottom_left":
                # Left side, near bottom
                x = 20
                y = self.main_window.height() - widget.height() - 20
                widget.move(x, y)
            elif position == "bottom_right":
                # Right side, near bottom
                x = self.main_window.width() - widget.width() - 20
                y = self.main_window.height() - widget.height() - 20
                widget.move(x, y)
            elif isinstance(position, (tuple, list)) and len(position) == 2:
                # Regular absolute position
                widget.move(*position)

            # Set word wrap
            if widget_config.get("word_wrap", False):
                widget.setWordWrap(True)

            # Set default text
            if "default_text" in widget_config:
                widget.setText(widget_config["default_text"])

            # Make label ignore mouse events if configured
            if widget_config.get("transparent_for_mouse", True):
                widget.setAttribute(Qt.WA_TransparentForMouseEvents)

            # Raise widget to top
            widget.raise_()

            # Store reference in the overlay_widgets dictionary
            widget_id = f"{widget_type}_{id(widget)}"
            self.overlay_widgets[widget_id] = {
                "widget": widget,
                "type": widget_type,
                "config": widget_config
            }

            # For backward compatibility
            if widget_type == "status":
                self.status_label = widget
            elif widget_type == "user_speech":
                self.user_speech_label = widget
            elif widget_type == "ai_response":
                self.ai_response_label = widget

            # Connect signals if defined in config
            if "signal" in widget_config and hasattr(self, 'signals'):
                signal_name = widget_config["signal"]
                signal = self.signals.get_signal(signal_name)
                if not signal:
                    signal = self.signals.create_signal(signal_name)
                signal.connect(lambda text: self._update_widget_text(widget, text))

            # Show the widget
            widget.show()

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


        return {
            "status": self.status_label if hasattr(self, "status_label") else None,
            "user_speech": self.user_speech_label if hasattr(self, "user_speech_label") else None,
            "ai_response": self.ai_response_label if hasattr(self, "ai_response_label") else None
        }

    def update_status(self, status_text, is_online=None):
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
        """Update the user speech display with the given text"""
        self.logger.debug(f"Updating user speech: '{text}'")

        if not hasattr(self, 'user_speech_label'):
            self.logger.warning("Cannot update user speech: user_speech_label not initialized")
            return False

        try:
            self.user_speech_label.setText(f"{text}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update user speech: {str(e)}")
            return False

    def update_ai_response(self, text):
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
            widget_instance (QWidget): Widget instance or widget_type string to update

        Returns:
            bool: True if successful, False otherwise
        """
        # If widget_instance is a string, try to find the widget by type
        if isinstance(widget_instance, str):
            found_widget = None
            for widget_id, info in self.overlay_widgets.items():
                if info["type"] == widget_instance:
                    found_widget = info["widget"]
                    break

            if found_widget:
                widget_instance = found_widget
            else:
                self.logger.warning(f"No widget found with type: {widget_instance}")
                return False

        return self._update_widget_text(widget_instance, text)

    def hide_widget(self, widget_instance):
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
        """
        name = task_name or task_func.__name__
        self.logger.debug(f"Creating worker thread for task: {name}")

        try:
            # Use the imported create_worker function
            worker = create_worker(
                task_func,
                *args,
                task_name=name,
                worker_type="qt"
            )
            worker.finished_signal.connect(lambda: self.cleanup_worker(worker))
            self.worker_threads.append(worker)
            return worker
        except Exception as e:
            self.logger.error(f"Failed to create worker thread: {str(e)}")
            return None

    def resizeEvent(self, event):
        """Handle window resize events to maintain widget positions"""
        super().resizeEvent(event)
        self.adjust_overlay_positions()

    def cleanup_worker(self, worker):
        """Remove a worker thread from the tracking list when it finishes"""
        if worker in self.worker_threads:
            task_name = getattr(worker, 'task_name', 'unknown')
            self.logger.debug(f"Cleaning up worker thread: {task_name}")
            self.worker_threads.remove(worker)

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
            return

        for widget_id, widget_info in self.overlay_widgets.items():
            widget = widget_info["widget"]
            position = widget_info["config"].get("position")

            if position == "center_top":
                x = (self.main_window.width() - widget.width()) // 2
                y = 20
                widget.move(x, y)
            elif position == "bottom_left":
                x = 20
                y = self.main_window.height() - widget.height() - 20
                widget.move(x, y)
            elif position == "bottom_right":
                x = self.main_window.width() - widget.width() - 20
                y = self.main_window.height() - widget.height() - 20
                widget.move(x, y)

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

    def _update_widget_text(self, widget_instance, text):
        """
        Universal method to update any widget's text

        Args:
            widget_instance: The widget to update
            text (str): Text to display

        Returns:
            bool: True if successful, False otherwise
        """
        if not widget_instance:
            self.logger.warning("Cannot update widget: widget_instance is None")
            return False

        try:
            # Check if this is a regular QLabel or similar widget
            if hasattr(widget_instance, 'setText'):
                widget_instance.setText(text)
                return True

            # Handle any special widget types here
            # ...

            self.logger.warning(f"Don't know how to update widget of type {type(widget_instance)}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to update widget text: {str(e)}")
            return False
