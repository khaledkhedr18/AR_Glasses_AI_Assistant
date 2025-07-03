import os
import importlib.util
from .Qt_Handler import QtHandler

class GUIHandler:
    """
    Main handler for managing GUI interfaces.

    This class serves as a factory and manager for different GUI implementations,
    allowing the application to switch between different GUI frameworks (Qt or Tkinter)
    while maintaining a consistent interface.
    """

    def __init__(self, gui_type="qt"):
        """
        Initialize the GUI Handler with the specified GUI type.

        Args:
            gui_type (str): The type of GUI to use - "qt" (default) or "tkinter"
        """
        self.gui_handler = None
        self.gui_type = gui_type.lower()

        # Load the appropriate GUI handler
        self._load_gui_handler()

    def create_overlay_widget(self,widget_type, config):
        """
        Create an overlay widget using the provided configuration

        Args:
            config (dict): Configuration for the widget including 'type'

        Returns:
            The created widget instance
        """
        if not config or 'type' not in config:
            print("Error: Widget configuration must include a 'type'")
            return None

        if self.gui_handler:
            return self.gui_handler.create_overlay_widget(config['type'], config)
        return None

    def create_overlay_widgets(self):
        """
        Create an overlay widget using the provided configuration

        Args:
            config (dict): Configuration for the widget including 'type'

        Returns:
            The created widget instance
        """

        if self.gui_handler:
            return self.gui_handler.create_overlay_widgets()
        return None

    def hide_widget(self, widget_instance):
        """
        Hide a widget

        Args:
            widget_instance: The widget to hide

        Returns:
            bool: True if successful, False otherwise
        """
        if self.gui_handler:
            return self.gui_handler.hide_widget(widget_instance)
        return False

    def hide_all_widgets(self):
        """
        Hide all overlay widgets

        Returns:
            bool: True if successful, False otherwise
        """
        if self.gui_handler:
            return self.gui_handler.hide_all_widgets()
        return False

    def show_widget(self, widget_instance):
        """
        Show a previously hidden widget

        Args:
            widget_instance: The widget to show

        Returns:
            bool: True if successful, False otherwise
        """
        if self.gui_handler:
            return self.gui_handler.show_widget(widget_instance)
        return False

    def display_text_in_widget(self, text, widget_instance):
        """
        Display text in a widget

        Args:
            text (str): The text to display
            widget_instance: The widget to update

        Returns:
            bool: True if successful, False otherwise
        """
        if self.gui_handler:
            return self.gui_handler.display_text_in_widget(text, widget_instance)
        return False

    def display_image_in_widget(self, image_path, widget_instance):
        """
        Display an image in a widget

        Args:
            image_path (str): Path to the image file
            widget_instance: The widget to update

        Returns:
            bool: True if successful, False otherwise
        """
        if self.gui_handler:
            return self.gui_handler.display_image_in_widget(image_path, widget_instance)
        return False


    def create_window(self, title="AR Glasses Assistant", fullscreen=True):
        """
        Create the main application window

        Args:
            title (str): Window title
            fullscreen (bool): Whether to show in fullscreen mode
        """
        if self.gui_handler:
            return self.gui_handler.create_window(title, fullscreen)
        return None

    def update_status(self, status_text, is_online=None):
        """
        Update the status display

        Args:
            status_text (str): Status text to display
            is_online (bool, optional): If provided, adds ONLINE/OFFLINE indicator
        """
        if self.gui_handler:
            self.gui_handler.update_status(status_text, is_online)

    def update_user_speech(self, text):
        """
        Update the user speech display

        Args:
            text (str): User speech text
        """
        if self.gui_handler:
            self.gui_handler.update_user_speech(text)

    def update_ai_response(self, text):
        """
        Update the AI response display

        Args:
            text (str): AI response text
        """
        if self.gui_handler:
            self.gui_handler.update_ai_response(text)

    def create_worker(self, task_func, *args, on_result=None, on_finished=None):
        """
        Create a worker thread for background tasks

        Args:
            task_func: Function to run
            *args: Arguments to pass to the function
            on_result: Callback for when task produces a result
            on_finished: Callback for when task completes

        Returns:
            A worker thread instance appropriate for the current GUI framework
        """
        from utils.WorkerThread import create_worker

        # Use the appropriate worker type based on the GUI framework
        worker_type = "thread"
        if self.gui_type == "qt":
            worker_type = "qt"

        worker = create_worker(
            task_func,
            *args,
            on_result=on_result,
            on_finished=on_finished,
            worker_type=worker_type
        )
        return worker

    def get_camera_widget(self):
        """
        Get the camera widget instance

        Returns:
            The camera widget from the active GUI handler
        """
        if self.gui_handler:
            return self.gui_handler.get_camera_widget()
        return None

    def get_signals(self):
        """
        Get the communication signals object

        Returns:
            Communication signals object from the active GUI handler
        """
        if self.gui_handler:
            if hasattr(self.gui_handler, 'signals'):
                return self.gui_handler.signals
        return None

    def run(self):
        """Run the application main loop"""
        if self.gui_handler:
            return self.gui_handler.run()
        return 1  # Error code

    def cleanup(self):
        """Clean up resources before exit"""
        if self.gui_handler:
            self.gui_handler.cleanup()

    def update_camera_frame(self, frame):
        """
        Update the camera display with a raw frame

        Args:
            frame: Raw camera frame data (numpy array)

        Returns:
            bool: True if successful, False otherwise
        """
        if self.gui_handler:
            return self.gui_handler.update_camera_frame(frame)
        return False

    def delete_overlay_widget(self, widget_instance):
        """
        Permanently delete an overlay widget

        Args:
            widget_instance: The widget to delete

        Returns:
            bool: True if successful, False otherwise
        """
        if self.gui_handler and hasattr(self.gui_handler, 'delete_overlay_widget'):
            return self.gui_handler.delete_overlay_widget(widget_instance)
        return False

    def _load_gui_handler(self):
        """Load the Qt GUI handler directly without fallbacks"""
        from utils.Logging import Logger
        # Initialize logger if needed
        logger = Logger()

        try:
            # Direct import since the file is in the same directory
            logger.info("Importing Qt Handler directly")

            # Create the Qt handler
            self.gui_handler = QtHandler()
            logger.info("Qt GUI handler loaded successfully")
        except ImportError as e:
            logger.error(f"Error importing Qt Handler: {e}")
            raise ImportError(f"Failed to import Qt_Handler: {e}")
        except Exception as e:
            logger.error(f"Critical error loading GUI handler: {e}")
            raise
