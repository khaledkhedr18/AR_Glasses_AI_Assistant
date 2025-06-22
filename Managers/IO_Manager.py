import os
import threading
import time
from Handlers import GUIHandler
from Handlers import CameraHandler
from Handlers import AudioHandler
from utils.Logging import Logger
from utils.Config import IO_CONFIG
from utils.Services import Services

class IOManager:
    def __init__(self):
        self.gui = GUIHandler()
        self.camera = CameraHandler()
        self.audio = AudioHandler()
        self.recognizer = Services(IO_CONFIG['RECOGNIZER_MODEL_PATH'])
        self.logger = Logger()
        self.wake_word = IO_CONFIG['WAKE_WORD']
        self.supported_languages = IO_CONFIG['SUPPORTED_LANGUAGES']
        self.camera_running = False
        self.camera_thread = None
        self.frame_captured = None
        self.audio_running = False
        self.recorded_audio = None
        self.camera_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.interaction_lock = threading.Lock()
        self.frame_interval = IO_CONFIG.get('FRAME_INTERVAL', 0.03)
        self.config = IO_CONFIG

    def start_camera_stream(self):
        """
        Start camera streaming in a separate thread.

        This method initializes the camera and starts a thread that
        continuously captures frames and updates the GUI.

        Returns:
            bool: True if the stream was started, False otherwise
        """
        with self.camera_lock:
            if self.camera_running and self.camera_thread and self.camera_thread.is_alive():
                self.logger.info("Camera stream already running")
                return False

            # Initialize the camera if needed
            if not self.initialize_camera():
                self.logger.error("Failed to initialize camera")
                return False

            self.camera_running = True
            self.camera_thread = threading.Thread(target=self._stream_camera_frames)
            self.camera_thread.daemon = True
            self.logger.info("Starting camera stream thread")
            self.camera_thread.start()

    def _stream_camera_frames(self):
        """
        Private method to handle continuous camera streaming.

        This method captures frames from the camera and forwards them
        to the GUI for display.
        """
        self.logger.info("Camera streaming thread started")
        frame_count = 0
        start_time = time.time()

        while self.camera_running:
            try:
                # Capture a frame from camera handler
                frame = self.camera.capture_frame()

                if frame is not None:
                    # Forward the frame to GUIHandler
                    self.gui.update_camera_frame(frame)
                    frame_count += 1

                # Control frame rate
                time.sleep(self.frame_interval)

            except Exception as e:
                self.logger.error(f"Error in camera streaming thread: {e}")
                if self.camera_running:
                    # Log FPS before exiting due to error
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    self.logger.info(f"Camera stream ended with error. Processed {frame_count} frames at {fps:.2f} FPS")
                break

        # Log performance metrics when stopping normally
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        self.logger.info(f"Camera stream stopped. Processed {frame_count} frames at {fps:.2f} FPS")


    def stop_camera_stream(self):
        """
        Stop the camera streaming thread.

        Returns:
            bool: True if the stream was stopped, False otherwise
        """
        with self.camera_lock:
            if not self.camera_running:
                self.logger.info("Camera stream already stopped")
                return True

            self.logger.info("Stopping camera stream")
            self.camera_running = False

            # Wait for the thread to terminate
            if self.camera_thread and self.camera_thread.is_alive():
                timeout = self.config.get('CAMERA_THREAD_TIMEOUT', 2.0)
                self.camera_thread.join(timeout=timeout)

            self.camera_thread = None
            return True

    def get_image(self):
        with self.camera_lock:
            if self.camera_running:
                return self.camera.capture_and_save_image()
            self.logger.warning("Camera not running, cannot capture image")
            return None

    def start_audio_listening(self, mode="offline"):
        """
        Start listening for audio input.

        Args:
            mode (str): "online" to save to file, "offline" to keep in memory

        Returns:
            bool: True if started successfully
        """
        with self.audio_lock:
            if not self.audio_running:
                self.logger.info(f"Starting audio recording in {mode} mode")
                success = self.audio.start_recording(mode)
                if success:
                    self.audio_running = True
                return success
            return False

    def stop_audio_listening(self):
        """
        Stop listening for audio input.

        Returns:
            Audio data or file path depending on recording mode
        """
        with self.audio_lock:
            if self.audio_running:
                self.logger.info("Stopping audio recording")
                audio_result = self.audio.stop_recording()
                self.audio_running = False
                return audio_result
            return None

    def get_user_audio(self, mode="offline", duration=5):
        """
        Get user audio synchronously.

        Args:
            mode (str): "online" to save to file, "offline" to return data
            duration (int): Recording duration in seconds

        Returns:
            Audio data (offline) or file path (online)
        """
        with self.audio_lock:
            if self.audio_running:
                self.logger.warning("Audio recording already in progress")
                return None

            # Start recording
            if not self.start_audio_listening(mode):
                return None

            # Wait for the specified duration
            time.sleep(duration)

            # Stop recording and get the result
            return self.stop_audio_listening()

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
                    self.gui.display_text_in_widget(IO_CONFIG['AI_WINDOW_NAME'], text)

            except Exception as e:
                self.logger.log_error_with_traceback("Error in user interaction", e)

    def get_user_config(self):
        """
        Interactive configuration through voice conversation with AI agent
        Returns: dict with source_lang, dest_lang, and translation_mode
        """

        config = {
            'source_lang': None,
            'dest_lang': None,
            'translation_mode': None
        }

        def wait_for_wake_word():
            prompt = f"Please say the wake word '{IO_CONFIG['WAKE_WORD']}' to start configuration."
            self.audio.output_speech(prompt)
            self.logger.info(prompt)
            self.gui.display_text_in_widget(IO_CONFIG['AI_WINDOW_NAME'], prompt)
            self.recorded_audio = self.__record_with_timer(IO_CONFIG['AUDIO_RECORD_TIMEOUT'])
            while True:
                text = self.recognizer.recognize_text_from_speech(self.recorded_audio)
                if text and IO_CONFIG['WAKE_WORD'] in text:
                    return True
                time.sleep(0.1)

        def get_language_input(prompt):
            self.audio.output_speech(prompt)
            self.logger.info(prompt)
            self.gui.display_text_in_widget(IO_CONFIG['AI_WINDOW_NAME'], prompt)
            self.recorded_audio = self.__record_with_timer(IO_CONFIG['AUDIO_RECORD_TIMEOUT'])
            while True:
                text = self.recognizer.recognize_text_from_speech(self.recorded_audio)
                if text:
                    for lang, code in IO_CONFIG['SUPPORTED_LANGUAGES'].items():
                        if lang in text:
                            return code
                time.sleep(0.1)

        def get_translation_mode():
            prompt = "What do you want to translate? (speech, image, or image with prompt)"
            self.audio.output_speech(prompt)
            self.logger.info(prompt)
            self.gui.display_text_in_widget(IO_CONFIG['AI_WINDOW_NAME'], prompt)
            self.recorded_audio = self.__record_with_timer(IO_CONFIG['AUDIO_RECORD_TIMEOUT'])
            while True:
                text = self.recognizer.recognize_text_from_speech(self.recorded_audio)
                if text:
                    if 'image' in text and 'speech' in text:
                        return 'both'
                    elif 'image' in text:
                        return 'image'
                    elif 'speech' in text:
                        return 'speech'
                time.sleep(0.1)

        # Start conversation flow
        if wait_for_wake_word():
            message = "Wake word detected! Starting configuration..."
            self.logger.info(message)
            self.gui.display_text_in_widget(IO_CONFIG['AI_WINDOW_NAME'], message)
            self.audio.output_speech(message)

            # Get source language
            config['source_lang'] = get_language_input("What is the source language?")

            # Get target language
            config['dest_lang'] = get_language_input("What is the target language?")

            # Get translation mode
            config['translation_mode'] = get_translation_mode()

            # Display final configuration
            final_config = f"Configuration set:\nFrom: {config['source_lang']}\nTo: {config['dest_lang']}\nMode: {config['translation_mode']}"
            self.logger.info(final_config)
            self.gui.display_text_in_widget(IO_CONFIG['AI_WINDOW_NAME'], final_config)
            self.audio.output_speech(final_config)

            return config if all(config.values()) else None

        return None

    def capture_frame(self):
        """
        Capture a single frame from the camera.

        Returns:
            numpy.ndarray: The captured frame
        """
        return self.camera.capture_frame()

    def capture_image(self, filename=None):
        """
        Capture an image and save it to a file.

        Args:
            filename (str, optional): Path to save the image

        Returns:
            str: Path to the saved image file
        """
        self.logger.info(f"IO_Manager: Capturing image to {filename or 'auto-generated file'}")
        if hasattr(self.camera, 'capture_image'):
            return self.camera.capture_image(filename)
        elif hasattr(self.camera, 'capture_and_save_image'):
            return self.camera.capture_and_save_image()
        return None

    def initialize_camera(self):
        """
        Initialize and start the camera.

        Returns:
            bool: True if the camera was initialized successfully, False otherwise
        """
        self.logger.info("IO_Manager: Initializing camera")
        if hasattr(self.camera, 'initialize_camera'):
            return self.camera.initialize_camera()
        return True  # Assume camera is ready if no initialize method

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
        if hasattr(self.camera, 'process_image'):
            return self.camera.process_image(image, processing_level)
        return image

    def set_camera_parameters(self, **kwargs):
        """Set camera parameters (exposure, gain, focus mode)."""
        self.logger.info(f"IO_Manager: Setting camera parameters: {kwargs}")
        if hasattr(self.camera, 'set_camera_parameters'):
            return self.camera.set_camera_parameters(**kwargs)
        elif hasattr(self.camera, 'set_camera_configurations'):
            return self.camera.set_camera_configurations(**kwargs)
        return False

    def get_user_command(self):
        """
        Records audio and determines the command from user's voice input
        Returns: str - The command to execute ('start', 'stop', 'translate', 'exit')
        """

        try:
            # Record audio for 5 seconds
            self.recorded_audio = self.__record_with_timer(5)
            if not self.recorded_audio:
                return None

            # Convert speech to text
            text = self.recognizer.recognize_text_from_speech(self.recorded_audio)
            if not text:
                return None

            # Convert to lowercase for better matching
            text = text.lower()
            self.logger.info(f"Recognized command: {text}")

            # Check for command keywords
            for command, keywords in IO_CONFIG['COMMAND_KEYWORDS'].items():
                if any(keyword in text for keyword in keywords):
                    return command

            return None

        except Exception as e:
            self.logger.info(f"Error processing command: {e}")
            return None
    def create_window(self, title="AR Glasses Assistant", fullscreen=True):
        """Create the main application window."""
        window = self.gui.create_window(title=title, fullscreen=fullscreen)
        return window

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

    # Add this new method to the IOManager class
    def delete_overlay_widget(self, widget_instance):
        """
        Permanently delete an overlay widget from the GUI.

        Args:
            widget_instance: The widget to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self.logger.info("Deleting overlay widget")
        return self.gui.delete_overlay_widget(widget_instance)
