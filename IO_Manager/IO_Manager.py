from utils.logging import Logger
from IO_Manager.IO_Handlers.Audio_Handler import AudioHandler
from IO_Manager.IO_Handlers.Camera_Handler import CameraHandler
from IO_Manager.GUI_Handlers.GUI_Handler import GUI_Handler

class IO_Manager:
    """
    Central communication hub for all IO operations.

    This class manages all handlers and provides interfaces for:
    1. Handlers to communicate with each other indirectly
    2. External managers (like LLM_Manager) to access IO functionality
    """

    def __init__(self, config=None):
        """Initialize the IO_Manager with all required handlers."""
        self.config = config or {}
        self.logger = Logger()
        self.logger.info("Initializing IO_Manager")

        # Initialize handlers
        self.audio_handler = AudioHandler()
        self.gui_handler = GUI_Handler(gui_type=self.config.get('gui_type', 'qt'))
        self.camera_handler = CameraHandler()

        # Track active workers
        self.active_workers = []

    # ======== INTERFACE METHODS FOR EXTERNAL MANAGERS ========

    def create_window(self, title="AR Glasses Assistant", fullscreen=True):
        """Create the main application window."""
        window = self.gui_handler.create_window(title=title, fullscreen=fullscreen)
        return window

    def display_text(self, text, is_user=False):
        """
        Display text on the GUI.

        Args:
            text (str): Text to display
            is_user (bool): If True, displays as user text, otherwise as AI response
        """
        if is_user:
            self.gui_handler.update_user_speech(text)
        else:
            self.gui_handler.update_ai_response(text)

    def update_status(self, status, is_online=None):
        """Update status display on the GUI."""
        self.gui_handler.update_status(status, is_online)

    # ======== AUDIO INTERFACE METHODS ========

    def start_listening(self, model, duration=10, on_result=None):
        """
        Start listening for audio input.

        Args:
            model: Speech recognition model
            duration (int): Maximum duration to listen for
            on_result: Callback for speech recognition result

        Returns:
            The worker handling the recording task
        """
        self.logger.info("IO_Manager: Starting audio listening")
        worker = self.audio_handler.start_recording(model, duration, on_result)
        if worker:
            self.active_workers.append(worker)
        return worker

    def stop_listening(self):
        """Stop listening for audio input."""
        self.logger.info("IO_Manager: Stopping audio listening")
        return self.audio_handler.stop_recording()

    def speak_text(self, text, on_finished=None):
        """
        Output text as speech.

        Args:
            text (str): Text to speak
            on_finished: Callback when speech finishes

        Returns:
            The worker handling the speech task
        """
        self.logger.info(f"IO_Manager: Speaking text: '{text}'")
        worker = self.audio_handler.output_speech(text, on_finished)
        if worker:
            self.active_workers.append(worker)
        return worker

    def mute_speech(self):
        """Stop any ongoing speech output."""
        self.logger.info("IO_Manager: Muting speech")
        return self.audio_handler.mute_speech()

    def is_listening(self):
        """Check if currently listening for audio input."""
        return self.audio_handler.is_listening()

    def get_user_audio(self, model, duration=10):
        """
        Get audio input from user (synchronous).

        Args:
            model: Speech recognition model
            duration (int): Maximum duration to listen for

        Returns:
            str: Recognized text from user speech
        """
        self.logger.info(f"IO_Manager: Getting user audio (max {duration}s)")
        return self.audio_handler.get_user_audio(model, duration)

    # ======== CAMERA INTERFACE METHODS ========

    def start_camera(self):
        """Initialize and start the camera."""
        self.logger.info("IO_Manager: Starting camera")
        return self.camera_handler.initialize_camera()

    def capture_frame(self):
        """
        Capture a single frame from the camera.

        Returns:
            numpy.ndarray: The captured frame
        """
        return self.camera_handler.capture_frame()

    def capture_image(self, filename=None):
        """
        Capture an image and save it to a file.

        Args:
            filename (str, optional): Path to save the image

        Returns:
            str: Path to the saved image file
        """
        self.logger.info(f"IO_Manager: Capturing image to {filename or 'auto-generated file'}")
        return self.camera_handler.capture_image(filename)

    def process_image(self, image, processing_level="medium"):
        """
        Process an image for better text recognition.

        Args:
            image: Image path or array
            processing_level (str): "low", "medium", or "high"

        Returns:
            numpy.ndarray: Processed image
        """
        self.logger.info(f"IO_Manager: Processing image with level '{processing_level}'")
        return self.camera_handler.process_image(image, processing_level)

    def set_camera_parameters(self, **kwargs):
        """Set camera parameters (exposure, gain, focus mode)."""
        self.logger.info(f"IO_Manager: Setting camera parameters: {kwargs}")
        return self.camera_handler.set_camera_parameters(**kwargs)

    def start_recording(self, output_file="video.mp4", fps=30, duration=None):
        """
        Start recording video.

        Args:
            output_file (str): Path to save the video
            fps (int): Frames per second
            duration (float, optional): Recording duration in seconds

        Returns:
            bool: True if recording started successfully
        """
        self.logger.info(f"IO_Manager: Starting video recording to {output_file}")
        return self.camera_handler.start_recording(output_file, fps, duration)

    def stop_recording(self):
        """Stop video recording."""
        self.logger.info("IO_Manager: Stopping video recording")
        return self.camera_handler.stop_recording()

    # ======== WORKER THREAD MANAGEMENT ========

    def create_worker(self, task_func, *args, on_result=None, on_finished=None):
        """
        Create a worker thread for background tasks.

        Args:
            task_func: Function to run in background
            *args: Arguments for the function
            on_result: Callback for when results are available
            on_finished: Callback for when task completes

        Returns:
            The created worker thread
        """
        worker = self.gui_handler.create_worker(
            task_func,
            *args,
            on_result=on_result,
            on_finished=on_finished
        )
        if worker:
            self.active_workers.append(worker)
        return worker

    # ======== APPLICATION CONTROL ========

    def run(self):
        """
        Run the application main loop.

        Returns:
            int: Application exit code
        """
        return self.gui_handler.run()

    def cleanup(self):
        """Clean up all resources before application exit."""
        self.logger.info("IO_Manager: Cleaning up resources")

        # Clean up handlers
        if hasattr(self, 'camera_handler'):
            self.camera_handler.cleanup()

        if hasattr(self, 'audio_handler'):
            self.audio_handler.cleanup()

        if hasattr(self, 'gui_handler'):
            self.gui_handler.cleanup()

        # Clear any remaining worker references
        self.active_workers.clear()
