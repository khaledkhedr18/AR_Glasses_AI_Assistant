from Handlers.GUI_Handler import GUIHandler
from Handlers.Camera_Handler import CameraHandler
from Handlers.Audio_Handler import AudioHandler
from vosk import Model, KaldiRecognizer
import threading

class IOManager:
    def __init__(self):
        self.gui = GUIHandler()
        self.camera = CameraHandler()
        self.audio = AudioHandler()
        self.camera_running = False
        self.image = None
        self.audio_running = False
        self.recorded_audio = None
        self.camera_lock = threading.Lock()
        self.audio_lock = threading.Lock()

    def start_camera_stream(self):
        with self.camera_lock:
            if not self.camera_running:
                self.image = self.camera.capture_image()
                self.gui.display_image_in_window(self.image)
                self.camera_running = True

    def stop_camera_stream(self):
        with self.camera_lock:
            if self.camera_running:
                self.gui.hide_window()
                self.camera_running = False

    def get_frame(self):
        with self.camera_lock:
            if self.camera_running:
                return self.camera.capture_image()
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
            while True:
                with self.audio_lock:
                    self.audio.start_recording()
                    audio_data = self.audio.stop_recording()
                    text = self.audio.get_vosk_recognition(audio_data)
                    if text and 'hi david' in text.lower():
                        return True

        def get_language_input(prompt):
            self.gui.display_text_in_window(prompt)
            while True:
                with self.audio_lock:
                    self.audio.start_recording()
                    audio_data = self.audio.stop_recording()
                    text = self.audio.get_vosk_recognition(audio_data)
                    if text:
                        text = text.lower()
                        for lang in lang_map:
                            if lang in text:
                                return lang_map[lang]
            return None

        def get_translation_mode():
            prompt = "What do you want to translate? (image, speech, or image with speech)"
            self.gui.display_text_in_window(prompt)
            while True:
                with self.audio_lock:
                    self.audio.start_recording()
                    audio_data = self.audio.stop_recording()
                    text = self.audio.get_vosk_recognition(audio_data)
                    if text:
                        text = text.lower()
                        if 'image' in text and 'speech' in text:
                            return 'both'
                        elif 'image' in text:
                            return 'image'
                        elif 'speech' in text:
                            return 'speech'
            return None

        # Start conversation flow
        if wait_for_wake_word():
            # Get source language
            config['source_lang'] = get_language_input("What is the source language?")

            # Get target language
            config['dest_lang'] = get_language_input("What is the target language?")

            # Get translation mode
            config['translation_mode'] = get_translation_mode()

            # Display final configuration
            self.gui.display_text_in_window(
                f"Configuration set:\nFrom: {config['source_lang']}\nTo: {config['dest_lang']}\nMode: {config['translation_mode']}")

            return config if all(config.values()) else None

        return None

    def get_user_command(self):
        """
        Determines the command from user input through GUI or voice
        Returns: str - The command to execute ('start', 'stop', 'translate', 'exit')
        """
        command = None

        # First check GUI command
        gui_command = self.gui.get_command()
        if gui_command:
            return gui_command

        # If no GUI command, check voice command
        with self.audio_lock:
            if self.recorded_audio:
                audio_text = self.audio.convert_speech_to_text(self.recorded_audio)
                audio_text = audio_text.lower()

                # Map keywords to commands
                command_mapping = {
                    'start': ['start', 'begin', 'launch'],
                    'stop': ['stop', 'end', 'finish'],
                    'translate': ['translate', 'convert', 'change'],
                    'exit': ['exit', 'quit', 'close']
                }

                # Check for command keywords
                for cmd, keywords in command_mapping.items():
                    if any(keyword in audio_text for keyword in keywords):
                        command = cmd
                        break

        return command

    def display_text(self, text):
        # Updates GUI with text
        self.gui.display_text_in_window("ai window", text)

    def get_audio(model, period=10):
        """
        Records audio from the default microphone and attempts to recognize spoken text.
        Returns the recognized text, or None if no audio is detected.
        """
        with audio_lock:
            recognizer = KaldiRecognizer(model, 16000)
            recognized_text = ""
            start_time = time.time()

            def callback(indata, frames, time, status):
                nonlocal recognized_text
                if recognizer.AcceptWaveform(indata.tobytes()):
                    result = json.loads(recognizer.Result())
                    recognized_text = result.get("text", "")
                    print(f"You said: {recognized_text}")

            print("Listening...")
            with sd.InputStream(callback=callback, channels=1, samplerate=16000, dtype=np.int16):
                while (time.time() - start_time) < period:
                    if recognized_text:
                        return recognized_text
                    sd.sleep(100)
            recognizer.Reset()
            return recognized_text