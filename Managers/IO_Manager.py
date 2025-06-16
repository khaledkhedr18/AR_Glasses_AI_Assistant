from IO_Handlers.GUI_Handler import GUIHandler
from IO_Handlers.Camera_Handler import CameraHandler
from IO_Handlers.Audio_Handler import AudioHandler
from Utils.Kaldi_Recognizer import KaldiRecognizer
import threading
import time


class IOManager:
    def __init__(self):
        self.gui = GUIHandler()
        self.camera = CameraHandler()
        self.audio = AudioHandler()
        self.recognizer = KaldiRecognizer()
        self.camera_running = False
        self.frame_captured = None
        self.audio_running = False
        self.recorded_audio = None
        self.wake_word = "hi david"             # config file
        self.camera_lock = threading.Lock()
        self.audio_lock = threading.Lock()

    def start_camera_stream(self):
        with self.camera_lock:
            if not self.camera_running:
                self.frame_captured = self.camera.capture_frame()
                self.gui.display_image_in_widget("camera window", self.frame_captured)
                self.camera_running = True

    def stop_camera_stream(self):
        with self.camera_lock:
            if self.camera_running:
                # stop camera thread
                self.gui.hide_widget("camera window")
                self.camera_running = False

    def get_image(self):
        with self.camera_lock:
            if self.camera_running:
                return self.camera.capture_and_save_image()
            return None

    def start_audio_listening(self):
        with self.audio_lock:
            if not self.audio_running:
                self.audio.start_recording()
                self.audio_running = True

    def stop_audio_listening(self):
        with self.audio_lock:
            if self.audio_running:
                self.recorded_audio = self.audio.stop_recording()
                self.audio_running = False

    def get_user_audio(self):
        with self.audio_lock:
            if self.audio_running:
                return self.recorded_audio
            return None

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

        # Language mapping
        lang_map = {
            'arabic': 'ar',
            'english': 'en',
            'french': 'fr',
            'spanish': 'es',
            'german': 'de'
        }

        def wait_for_wake_word():
            prompt = f"Please say the wake word '{self.wake_word}' to start configuration."
            self.audio.output_speech(prompt)
            print(prompt)
            self.gui.display_text_in_widget("ai window", prompt)
            self.recorded_audio = self.__record_with_timer(5)
            while True:
                text = self.recognizer.recognize_text_from_speech(self.recorded_audio)
                if text and self.wake_word in text:
                    return True
                time.sleep(0.1)

        def get_language_input(prompt):
            self.audio.output_speech(prompt)
            print(prompt)
            self.gui.display_text_in_widget(prompt)
            self.recorded_audio = self.__record_with_timer(5)
            while True:
                text = self.recognizer.recognize_text_from_speech(self.recorded_audio)
                if text:
                    for lang in lang_map:
                        if lang in text:
                            return lang_map[lang]
                time.sleep(0.1)

        def get_translation_mode():
            prompt = "What do you want to translate? (speech, image, or image with prompt)"
            self.audio.output_speech(prompt)
            print(prompt)
            self.gui.display_text_in_widget(prompt)
            self.recorded_audio = self.__record_with_timer(5)
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
            print("Wake word detected! Starting configuration...")
            self.gui.display_text_in_widget("Wake word detected! Starting configuration...")
            self.audio.output_speech("Wake word detected! Starting configuration...")

            # Get source language
            config['source_lang'] = get_language_input("What is the source language?")

            # Get target language
            config['dest_lang'] = get_language_input("What is the target language?")

            # Get translation mode
            config['translation_mode'] = get_translation_mode()

            # Display final configuration
            final_config = f"Configuration set:\nFrom: {config['source_lang']}\nTo: {config['dest_lang']}\nMode: {config['translation_mode']}"
            print(final_config)
            self.gui.display_text_in_widget(final_config)
            self.audio.output_speech(final_config)

            return config if all(config.values()) else None

        return None

    def get_user_command(self):
        """
        Records audio and determines the command from user's voice input
        Returns: str - The command to execute ('start', 'stop', 'translate', 'exit')
        """
        # Command keywords mapping
        command_mapping = {
            'start': ['start', 'begin', 'launch', 'activate', 'open'],
            'stop': ['stop', 'end', 'finish', 'quit'],
            'translate': ['translate', 'convert', 'change', 'interpret'],
            'exit': ['exit', 'quit', 'close', 'leave']
        }

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
            print(f"Recognized command: {text}")

            # Check for command keywords
            for command, keywords in command_mapping.items():
                if any(keyword in text for keyword in keywords):
                    return command

            return None

        except Exception as e:
            print(f"Error processing command: {e}")
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

                self.audio_running = False
                return recorded_audio[0]


