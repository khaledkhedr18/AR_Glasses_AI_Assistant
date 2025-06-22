from Handlers.GUI_Handler import GUIHandler
from Handlers.Camera_Handler import CameraHandler
from Handlers.Audio_Handler import AudioHandler
from utils.Services import Services
from utils.Config import IOConfig, ServicesConfig
import time
from utils.Logging import Logger
import threading


# This module handles all I/O operations including camera, audio, and GUI interactions.
class IOManager:
    def __init__(self):
        """
        Initializes the IO Manager with necessary components and configurations.
        """
        self.logger = Logger()
        self.logger.info("Initializing IO Handler")

        # Initialize handlers for GUI, Camera, Audio, and Services
        self.gui = GUIHandler()
        self.camera = CameraHandler()
        self.audio = AudioHandler()
        self.service = Services()

        # Initialize configuration parameters
        self.wake_word = IOConfig.INTERFACE['WAKE_WORD']
        self.mode_keywords = IOConfig.KEYWORDS['MODE']
        self.command_keywords = IOConfig.KEYWORDS['COMMANDS']
        self.audio_record_timeout = IOConfig.TIMING['AUDIO_RECORD_TIMEOUT']
        self.camera_thread_timeout = IOConfig.TIMING['CAMERA_THREAD_TIMEOUT']
        self.supported_languages = ServicesConfig.LANGUAGES['MAPPING']

        # Initialize state variables
        self.camera_running = False
        self.camera_thread = None
        self.frame_captured = None
        self.audio_running = False
        self.recorded_audio = None

        # Locks for thread safety
        self.camera_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.interaction_lock = threading.Lock()

    def start_camera_stream(self):
        """Starts camera stream in a separate thread with 30ms interval"""
        with self.camera_lock:
            if not self.camera_running:
                self.camera_running = True
                self.camera_thread = threading.Thread(target=self.__stream_camera)
                self.camera_thread.daemon = True
                self.logger.info("Starting camera stream thread")
                self.camera_thread.start()
            else:
                self.logger.info("Camera stream already running")

    def stop_camera_stream(self):
        """Stops the camera stream thread and hides the camera window"""
        with self.camera_lock:
            if self.camera_running:
                self.logger.info("Stopping camera stream")
                self.camera_running = False  # This will break the while loop in __stream_camera
                if self.camera_thread and self.camera_thread.is_alive():
                    self.camera_thread.join(timeout=IOConfig.TIMING['CAMERA_THREAD_TIMEOUT'])  # Wait up to configurable seconds for thread to finish
                    self.logger.info("Camera thread stopped")
                self.gui.delete_overlay_widget(IOConfig.INTERFACE['CAMERA'])
                self.camera_thread = None
                self.frame_captured = None
            else:
                self.logger.info("Camera stream already stopped")

    def get_image(self):
        with self.camera_lock:
            if self.camera_running:
                self.logger.warning("Camera not running, cannot capture image")
                return self.camera.capture_and_save_image()
            self.logger.warning("Camera not running, cannot capture image")
            return None

    def start_audio_listening(self):
        with self.audio_lock:
            if not self.audio_running:
                self.logger.info("Starting audio recording")
                self.audio.start_recording()
                self.audio_running = True

    def stop_audio_listening(self):
        with self.audio_lock:
            if self.audio_running:
                self.logger.info("Stopping audio recording")
                self.recorded_audio = self.audio.stop_recording()
                self.audio_running = False

    def get_user_audio(self):
        with self.audio_lock:
            if self.audio_running:
                return self.recorded_audio
            return None

    def interact_with_user(self, text, mode="both"):
        """
        Interacts with user through speech and display with thread safety.
        Args:
            text (str): Text to be spoken and displayed
            mode (str): Interaction mode - "speech", "display", or "both" (default)
        """
        with self.interaction_lock:  # Create a dedicated lock for interaction
            try:
                if mode in ["speech", "both"]:
                    self.audio.output_speech(text)

                if mode in ["display", "both"]:
                    self.logger.info(f"AI: {text}")
                    self.gui.display_text_in_widget(IOConfig.INTERFACE['AI'], text)

            except Exception as e:
                self.logger.log_error_with_traceback("Error in user interaction", e)

    def get_user_config(self):
        """
        Interactive configuration through voice conversation with AI agent
        Returns: dict with source_lang, dest_lang, and translation_mode
        """
        max_attempts = IOConfig.INTERFACE['MAX_ATTEMPTS']  # Maximum number of retry attempts

        config = {
            'source_lang': None,
            'dest_lang': None,
            'translation_mode': None
        }

        def wait_for_wake_word():
            """
            Waits for the user to say the wake word to start configuration
            """
            prompt = f"Please say the wake word '{IOConfig.INTERFACE['WAKE_WORD']}' to start configuration."
            self.logger.info(prompt)
            self.interact_with_user(prompt)

            self.recorded_audio = self.__record_with_timer(self.audio_record_timeout)
            if not self.recorded_audio:
                return None
            text = self.service.recognize_text_from_speech(self.recorded_audio)
            if text:
                return self.service.verify_user_input(text, self.wake_word) is not None

            return False

        def get_user_input(prompt):
            """
            Asks user for language input or translation mode (speech, image, or both) and returns the selected mode or recognized language code
            """
            self.logger.info(prompt)
            self.interact_with_user(prompt)

            self.recorded_audio = self.__record_with_timer(self.audio_record_timeout)
            if not self.recorded_audio:
                return None
            text = self.service.recognize_text_from_speech(self.recorded_audio)
            if text:
                return self.service.verify_user_input(text, self.mode_keywords)
            return None

        def retry_input(input_func, prompt, attempt=1):
            """Helper function to handle retries for input functions"""
            result = input_func(prompt)
            if not result and attempt < max_attempts:
                retry_message = f"Could not understand. Please try again. ({attempt + 1}/{max_attempts})"
                self.interact_with_user(retry_message)
                return retry_input(input_func, prompt, attempt + 1)
            elif not result:
                restart_message = "Maximum attempts reached. Please say the wake word to start over."
                self.interact_with_user(restart_message)
                return None
            return result

        # Start conversation flow
        if not wait_for_wake_word():
            return None

        message = "Wake word detected! Starting configuration..."
        self.logger.info(message)
        self.interact_with_user(message)

        # Get source language
        source_lang = retry_input(get_user_input, "What is the source language?")
        if not source_lang:
            self.logger.error("Failed to recognize source language after multiple attempts")
            return None
        config['source_lang'] = source_lang

        # Get target language
        dest_lang = retry_input(get_user_input, "What is the target language?")
        if not dest_lang:
            self.logger.error("Failed to recognize destination language after multiple attempts")
            return None
        config['dest_lang'] = dest_lang

        # Get translation mode
        translation_mode = retry_input(get_user_input, "What do you want to translate? (speech, image, or image with prompt)")
        if not translation_mode:
            self.logger.error("Failed to recognize translation mode after multiple attempts")
            return None
        config['translation_mode'] = translation_mode

        # Display final configuration
        final_config = f"Configuration set:\nFrom: {config['source_lang']}\nTo: {config['dest_lang']}\nMode: {config['translation_mode']}"
        self.logger.info(final_config)
        self.interact_with_user(final_config)
        return config if all(config.values()) else None

    def get_user_command(self):
        """
        Records audio and determines the command from user's voice input
        Returns: str - The command to execute ('start', 'stop', 'translate', 'exit')
        """

        try:
            # Record audio for 5 seconds
            self.recorded_audio = self.__record_with_timer(self.audio_record_timeout)
            if not self.recorded_audio:
                return None

            # Convert speech to text
            text = self.service.recognize_text_from_speech(self.recorded_audio)
            if not text:
                return None

            # Convert to lowercase for better matching
            text = text.lower()
            self.logger.info(f"Recognized command: {text}")

            # Check for command keywords
            for command, keywords in IOConfig.KEYWORDS['COMMANDS'].items():
                if any(keyword in text for keyword in keywords):
                    return command

            return None

        except Exception as e:
            self.logger.info(f"Error processing command: {e}")
            return None

    def display_text(self, label, text):
        # Updates GUI with text
        self.gui.display_text_in_widget(label, text)

    def __record_with_timer(self, timeout_seconds):
        """
        Records audio with timer in a separate thread.
        Ensures proper thread management and cleanup.

        Args:
            timeout_seconds (int): Duration to record in seconds
        Returns:
            recorded_audio: The recorded audio data or None if failed
        """
        with self.audio_lock:
            # Check if recording thread already exists and is running
            if hasattr(self, 'recording_thread') and self.recording_thread and self.recording_thread.is_alive():
                self.logger.warning("Audio recording thread already running")
                return None

            if self.audio_running:
                self.logger.warning("Audio recording already in progress")
                return None

            recorded_audio = [None]  # Using list as a mutable container
            recording_complete = threading.Event()

            def recording_task():
                try:
                    self.logger.info(f"Starting audio recording for {timeout_seconds} seconds")
                    self.audio.start_recording()

                    # Wait for the specified duration
                    if recording_complete.wait(timeout=timeout_seconds):
                        self.logger.info("Recording stopped before timeout")
                    else:
                        self.logger.info("Recording completed after timeout")

                    # Capture the recorded audio
                    recorded_audio[0] = self.audio.stop_recording()

                except Exception as e:
                    self.logger.log_error_with_traceback("Error during audio recording", e)
                    recorded_audio[0] = None
                finally:
                    self.audio_running = False
                    if hasattr(self, 'recording_thread'):
                        self.recording_thread = None
                    recording_complete.set()

            try:
                self.audio_running = True
                self.recording_thread = threading.Thread(target=recording_task)
                self.recording_thread.daemon = True
                self.recording_thread.start()

                # Wait for the recording to complete with a small buffer time
                self.recording_thread.join(timeout=timeout_seconds + 1)

                # Check if thread is still alive after timeout
                if self.recording_thread and self.recording_thread.is_alive():
                    self.logger.warning("Recording thread exceeded timeout, forcing stop")
                    recording_complete.set()
                    self.recording_thread.join(timeout=1)
                    if self.audio_running:
                        self.audio.stop_recording()
                        self.audio_running = False

                return recorded_audio[0]

            except Exception as e:
                self.logger.log_error_with_traceback("Error managing recording thread", e)
                if self.audio_running:
                    self.audio.stop_recording()
                    self.audio_running = False
                return None

    def __stream_camera(self):
        """Private method to handle continuous camera streaming"""
        while self.camera_running:  # Will stop when camera_running becomes False
            try:
                if not self.camera_running:  # Double check in case flag changed
                    break
                self.frame_captured = self.camera.capture_frame()
                self.gui.display_image_in_widget(IOConfig.INTERFACE['CAMERA'], self.frame_captured)
                time.sleep(IOConfig.TIMING['FRAME_INTERVAL'])  # 30ms interval
            except Exception as e:
                self.logger.log_error_with_traceback("Error in camera stream", e)
                self.camera_running = False  # Ensure we exit on error
                break
