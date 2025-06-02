from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
import sounddevice as sd
import numpy as np
import time
import json
import threading
from vosk import KaldiRecognizer

class TTSThread(QThread):
    finished = pyqtSignal()

    def __init__(self, text):
        """
        Initialize a TTSThread with given text.

        This constructor takes a text string as an argument and stores it
        as an instance variable. The text is played as speech when the
        thread is started.

        :param text: The text to play as speech.
        """
        super().__init__()
        self.text = text

    def run(self):
        """
        Runs the thread to play the given text as speech.

        This method is called automatically when the thread is started. It
        runs the flite command to play the given text as speech and then
        emits a finished signal.

        :return: None
        """
        subprocess.run(["flite", self.text])
        self.finished.emit()

class AudioHandler:
    def __init__(self):
        """
        Initialize the AudioHandler.

        Sets up the internal list to track TTS threads and audio lock for recording.
        """
        self.tts_threads = []
        self.audio_lock = threading.Lock()
        self.listening = False

    def speak(self, text):
        """
        Speak the given text using the flite text-to-speech engine.

        Args:
            text (str): The text to be spoken.

        Returns:
            None
        """
        clean_text = " ".join(str(text).splitlines()).strip()
        clean_text = clean_text.replace('"', '').replace("'", "")
        tts_thread = TTSThread(clean_text)

        def cleanup():
            self.remove_thread(tts_thread)

        tts_thread.finished.connect(cleanup)
        self.tts_threads.append(tts_thread)
        tts_thread.start()

    def remove_thread(self, thread):
        """
        Remove a thread from the internal tracking list.

        Args:
            thread: The thread to remove

        Returns:
            None
        """
        if thread in self.tts_threads:
            self.tts_threads.remove(thread)

    def get_audio(self, model, period=10):
        """
        Records audio from the default microphone and attempts to recognize spoken text.

        Args:
            model: The Vosk speech recognition model to use
            period (int): Maximum recording time in seconds

        Returns:
            str: The recognized text, or empty string if no speech detected

        Raises:
            Exception: If audio device cannot be accessed or other errors occur
        """
        with self.audio_lock:
            try:
                self.listening = True
                recognizer = KaldiRecognizer(model, 16000)
                recognized_text = ""
                start_time = time.time()

                def callback(indata, frames, time, status):
                    nonlocal recognized_text
                    if status:
                        print(f"Audio input error: {status}")

                    # Process audio data
                    if recognizer.AcceptWaveform(indata.tobytes()):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "")
                        if text.strip():  # Only update if we got non-empty text
                            recognized_text = text
                            print(f"You said: {recognized_text}")

                print("Listening...")
                with sd.InputStream(callback=callback, channels=1, samplerate=16000,
                                    dtype=np.int16, blocksize=8000):
                    # Wait until we either get text or timeout
                    while (time.time() - start_time) < period:
                        if recognized_text:
                            self.listening = False
                            return recognized_text
                        sd.sleep(100)

                # Get any partial results
                final_result = json.loads(recognizer.FinalResult())
                final_text = final_result.get("text", "")
                if final_text and not recognized_text:
                    recognized_text = final_text

                return recognized_text

            except Exception as e:
                print(f"Error recording audio: {e}")
                return ""
            finally:
                self.listening = False
                if 'recognizer' in locals():
                    recognizer.Reset()

    def is_listening(self):
        """
        Check if the handler is currently listening for audio input.

        Returns:
            bool: True if currently listening, False otherwise
        """
        return self.listening

    def cleanup(self):
        """
        Clean up resources used by the AudioHandler.

        Waits for all TTS threads to complete before returning.
        """
        # Wait for all threads to complete
        for thread in list(self.tts_threads):
            if thread.isRunning():
                thread.wait(1000)  # Wait up to 1 second
