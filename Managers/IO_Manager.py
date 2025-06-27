import threading
import time
from Handlers.GUI_Handler import GUIHandler
from Handlers.Camera_Handler import CameraHandler
from Handlers.Audio_Handler import AudioHandler
from utils.Logging import Logger
from utils.Config import IO_CONFIG, SERVICES_CONFIG
from utils.Services import Services


# This module handles all I/O operations including camera, audio, and GUI interactions.
class IOManager:
    def __init__(self, speech_model):
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
        self.__load_config()

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
        self.speech_model = speech_model

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
                    self.camera_thread.join(timeout=self.camera_thread_timeout)  # Wait up to configurable seconds for thread to finish
                    self.logger.info("Camera thread stopped")

                # show black screen in camera widget

                self.camera_thread = None
                self.frame_captured = None
            else:
                self.logger.info("Camera stream already stopped")

    def get_image(self):
        """Capture and return an image from the camera"""
        with self.camera_lock:
            if not self.camera_running:
                self.logger.warning("Camera not running, starting camera for capture")
                # Start camera if not already running
                self.start_camera_stream()
                time.sleep(0.5)  # Wait for camera to initialize

            self.logger.info("Capturing image from camera...")
            return self.camera.capture_and_save_image()

    def start_audio_listening(self):
        """Start audio recording if not already running"""
        with self.audio_lock:
            if self.audio_running:
                self.logger.debug("Audio already recording - not starting again")
                return

            # self.logger.info("Starting audio recording")
            success = self.audio.start_recording()
            if success:
                self.audio_running = True
            else:
                self.logger.warning("Failed to start audio recording")

    def stop_audio_listening(self):
        with self.audio_lock:
            if self.audio_running:
                # self.logger.info("Stopping audio recording")
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
        with self.interaction_lock:
            try:
                if mode in ["speech", "both"]:
                    self.logger.info(f"Speaking: {text}")
                    self.audio.output_speech(text)

                if mode in ["display", "both"]:
                    self.logger.info(f"AI: {text}")
                    # Update the AI response widget directly instead of using display_text_in_widget
                    self.gui.update_ai_response(text)
            except Exception as e:
                self.logger.log_error_with_traceback("Error in user interaction", e)

    def get_user_config(self):
        """
        Interactive configuration through voice conversation with AI agent
        Returns: dict with source_lang, dest_lang, and translation_mode
        """
        max_attempts = IO_CONFIG.get('INTERFACE', {}).get('MAX_ATTEMPTS', '')  # Maximum number of retry attempts

        config = {
            'source_lang': None,
            'dest_lang': None,
            'translation_mode': None
        }

        def __wait_for_wake_word():
            """
            Waits for the user to say the wake word to start configuration
            """
            prompt = f"Please say the wake word '{self.wake_word}' to start configuration."
            self.logger.info(prompt)
            self.interact_with_user(prompt)

            self.recorded_audio = self.__record_with_timer(self.audio_record_timeout)
            if not self.recorded_audio:
                return None
            text = self.service.recognize_text_from_speech(self.recorded_audio)
            if text:
                return self.service.verify_user_input(text, self.wake_word) is not None

            return False

        def __get_user_language(prompt):
            """
            Asks user for language input (source and target languages) and returns the recognized language code
            """
            self.logger.info(prompt)
            self.interact_with_user(prompt)

            self.recorded_audio = self.__record_with_timer(self.audio_record_timeout)
            if not self.recorded_audio:
                return None
            text = self.service.recognize_text_from_speech(self.recorded_audio)
            if text:
                return self.service.verify_user_input(text, self.supported_languages)
            return None

        def __get_user_mode(prompt):
            """
            Asks user for translation mode and returns the selected mode (speech, image, or both)
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

        def __retry_input(input_func, prompt, attempt=1):
            """Helper function to handle retries for input functions"""
            result = input_func(prompt)
            if not result and attempt < max_attempts:
                retry_message = f"Could not understand. Please try again. ({attempt + 1}/{max_attempts})"
                self.interact_with_user(retry_message)
                return __retry_input(input_func, prompt, attempt + 1)
            elif not result:
                restart_message = "Maximum attempts reached. Please say the wake word to start over."
                self.interact_with_user(restart_message)
                return None
            return result

        # Start conversation flow
        if not __wait_for_wake_word():
            return None

        message = "Wake word detected! Starting configuration..."
        self.logger.info(message)
        self.interact_with_user(message)

        # Get source language
        source_lang = __retry_input(__get_user_language, "What is the source language?")
        if not source_lang:
            self.logger.error("Failed to recognize source language after multiple attempts")
            return None
        config['source_lang'] = source_lang

        # Get target language
        dest_lang = __retry_input(__get_user_language, "What is the target language?")
        if not dest_lang:
            self.logger.error("Failed to recognize destination language after multiple attempts")
            return None
        config['dest_lang'] = dest_lang

        # Get translation mode
        translation_mode = __retry_input(__get_user_mode, "What do you want to translate? (speech, image, or image with prompt)")
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

            # Explicitly check for None instead of using "not" which can cause issues with numpy arrays
            if self.recorded_audio is None:
                return None

            # Convert speech to text
            text = self.service.recognize_text_from_speech(self.recorded_audio, self.speech_model)
            if text is None:
                return None

            # Convert to lowercase for better matching
            text = text.lower()
            self.logger.info(f"Recognized command: {text}")

            # IMPORTANT: Display the recognized text in the GUI
            self.gui.update_user_speech(f"You: {text}")

            # Check for command keywords with explicit boolean checks to avoid numpy array issues
            for command, keywords in self.command_keywords.items():
                # Safe comparison with explicit loop rather than "any()" to avoid boolean ambiguity
                match_found = False
                for keyword in keywords:
                    if keyword in text:
                        match_found = True
                        break

                if match_found:
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
                if self.frame_captured is not None:
                    # Use update_camera_frame instead of display_image_in_widget for raw frames
                    self.gui.update_camera_frame(self.frame_captured)
                time.sleep(self.frame_interval)  # 30ms interval
            except Exception as e:
                self.logger.log_error_with_traceback("Error in camera stream", e)
                self.camera_running = False  # Ensure we exit on error
                break

    def __load_config(self):
        self.wake_word = IO_CONFIG.get('USER_COMMAND', {}).get('WAKE_WORD', 'Hi David')
        self.supported_languages = SERVICES_CONFIG.get('LANGUAGES', {}).get('MAPPING', {})
        self.frame_interval = IO_CONFIG.get('FRAME_INTERVAL', 0.03)
        self.camera_thread_timeout = IO_CONFIG.get('CAMERA_THREAD_TIMEOUT', 2.0)
        self.audio_record_timeout = IO_CONFIG.get('TIMING', {}).get('AUDIO_RECORD_TIMEOUT', 5)
        self.mode_keywords = IO_CONFIG.get('USER_COMMANDS', {}).get('MODE', {})
        self.command_keywords = IO_CONFIG.get('USER_COMMANDS', {}).get('COMMANDS', {})

    def create_main_window(self, title="AR Glasses Assistant", fullscreen=True):
        """
        Creates the main application window

        Args:
            title (str): The window title
            fullscreen (bool): Whether to display in fullscreen mode

        Returns:
            QMainWindow: Reference to the created window
        """
        self.logger.info(f"Creating main window: title='{title}', fullscreen={fullscreen}")
        return self.gui.create_window(title, fullscreen)

    def create_all_overlay_widgets(self):
        """
        Creates all standard overlay widgets (status, user_speech, ai_response)

        Returns:
            dict: Dictionary with references to all created widgets
        """
        self.logger.info("Creating all standard overlay widgets")
        return self.gui.create_overlay_widgets()

    def create_custom_overlay_widget(self, widget_type, config=None):
        """
        Creates a custom overlay widget with specific type and configuration

        Args:
            widget_type (str): Type identifier for the widget
            config (dict, optional): Configuration parameters for the widget

        Returns:
            QWidget: Reference to the created widget
        """
        self.logger.debug(f"Creating custom overlay widget: {widget_type}")
        return self.gui.create_overlay_widget(widget_type, config)

    def show_overlay_widget(self, widget_instance):
        """
        Shows a previously hidden overlay widget

        Args:
            widget_instance: Widget reference to show

        Returns:
            bool: True if successful, False otherwise
        """
        return self.gui.show_widget(widget_instance)

    def hide_overlay_widget(self, widget_instance):
        """
        Hides an overlay widget

        Args:
            widget_instance: Widget reference to hide

        Returns:
            bool: True if successful, False otherwise
        """
        return self.gui.hide_widget(widget_instance)

    def delete_overlay_widget(self, widget_instance):
        """
        Permanently removes an overlay widget from the application

        Args:
            widget_instance: Widget reference to delete

        Returns:
            bool: True if successful, False otherwise
        """
        return self.gui.delete_overlay_widget(widget_instance)

    def get_user_speech(self, max_duration=5):
        """
        Record audio and perform real-time speech recognition.

        Args:
            max_duration (int): Maximum recording duration in seconds

        Returns:
            str: The recognized text or None if no speech detected
        """
        with self.audio_lock:
            if self.audio_running:
                self.logger.warning("Audio already recording - cannot start speech recognition")
                return None

            self.audio_running = True
            try:
                # Use the new method for real-time recognition
                text = self.audio.record_and_recognize(self.speech_model, max_duration)
                return text
            finally:
                self.audio_running = False

