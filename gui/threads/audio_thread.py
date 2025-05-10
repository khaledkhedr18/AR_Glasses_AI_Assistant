from PyQt5.QtCore import QThread, pyqtSignal
from core.models.base import ModelFactory
from config.settings import Settings

class AudioThread(QThread):
    """Thread for handling audio input and speech recognition"""

    text_received = pyqtSignal(str)

    def __init__(self, language_code: str = "en"):
        super().__init__()
        self.settings = Settings()
        self.speech_model = ModelFactory.get_model("speech", language_code=language_code)
        self._is_paused = False
        self._is_running = True

    def pause(self):
        """Pause audio processing"""
        self._is_paused = True

    def resume(self):
        """Resume audio processing"""
        self._is_paused = False

    def stop(self):
        """Stop the thread"""
        self._is_running = False
        self.quit()
        self.wait()

    def run(self):
        """Main thread loop"""
        while self._is_running:
            if not self._is_paused:
                # Record audio
                audio_data = self.speech_model.record_audio(duration=5.0)
                if audio_data:
                    # Recognize speech
                    text = self.speech_model.recognize(audio_data)
                    if text:
                        self.text_received.emit(text)
            self.msleep(100)  # Prevent CPU overload
