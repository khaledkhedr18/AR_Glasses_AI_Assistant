from Handlers.GUI_Handler import GUIHandler
from Handlers.Camera_Handler import CameraHandler
from Handlers.Audio_Handler import AudioHandler
from Utils.Kaldi_Recognizer import SpeechRecognizer
from Utils.Logging import Logger
from Utils.Config import IO_CONFIG
import threading
import time


class IOManager:
    def __init__(self):
        self.gui = GUIHandler()
        self.camera = CameraHandler()
        self.audio = AudioHandler()
        self.logger = Logger()
        self.wake_word = IO_CONFIG['WAKE_WORD']
        self.supported_languages = IO_CONFIG['SUPPORTED_LANGUAGES']
        self.recognizer = SpeechRecognizer(IO_CONFIG['RECOGNIZER_MODEL_PATH'])
        self.camera_running = False
        self.camera_thread = None
        self.frame_captured = None
        self.audio_running = False
        self.recorded_audio = None
        self.camera_lock = threading.Lock()
        self.audio_lock = threading.Lock()
        self.interaction_lock = threading.Lock()

    def start_camera_stream(self):
        """Starts camera stream in a separate thread with 30ms interval"""
        with self.camera_lock:
            if not self.camera_running and (not self.camera_thread or not self.camera_thread.is_alive()):
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
                    self.camera_thread.join(timeout=IO_CONFIG['CAMERA_THREAD_TIMEOUT'])  # Wait up to configurable seconds for thread to finish
                    self.logger.info("Camera thread stopped")
                self.gui.delete_overlay_widget(IO_CONFIG['CAMERA_WINDOW_NAME'])
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
                if IO_CONFIG['WAKE_WORD'] in text:
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

    def display_text(self, label, text):
        # Updates GUI with text
        self.gui.display_text_in_widget(label, text)

    def __record_with_timer(self, timeout_seconds):
        """
        Starts recording and sets a timer to stop after timeout_seconds
        Args:
            timeout_seconds (int): Seconds to record
        Returns:
            str: Path to the recorded audio file
        """
        with self.audio_lock:
            if not self.audio_running:
                self.logger.info(f"Starting timed recording for {timeout_seconds} seconds")
                self.audio_running = True
                recorded_event = threading.Event()
                recorded_audio = [None]  # Using list as a mutable container

                def stop_recording_timer():
                    recorded_audio[0] = self.audio.stop_recording()
                    recorded_event.set()

                self.audio.start_recording()
                timer = threading.Timer(timeout_seconds, stop_recording_timer)
                timer.start()

                # Wait for recording to complete
                recorded_event.wait(timeout=timeout_seconds + 1)  # Add 1 second buffer

                # Cancel timer if it hasn't fired yet
                if timer.is_alive():
                    timer.cancel()

                self.logger.info("Timed recording completed")
                self.audio_running = False
                return recorded_audio[0]

    def __stream_camera(self):
        """Private method to handle continuous camera streaming"""
        while self.camera_running:  # Will stop when camera_running becomes False
            try:
                if not self.camera_running:  # Double check in case flag changed
                    break
                self.frame_captured = self.camera.capture_frame()
                self.gui.display_image_in_widget(IO_CONFIG['CAMERA_WINDOW_NAME'], self.frame_captured)
                time.sleep(IO_CONFIG['FRAME_INTERVAL'])  # 30ms interval
            except Exception as e:
                self.logger.log_error_with_traceback("Error in camera stream", e)
                self.camera_running = False  # Ensure we exit on error
                break
