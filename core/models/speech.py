from typing import Optional, Dict, Any
import vosk
import wave
import json
from .base import BaseModel
from core.utils.logging import Logger

class SpeechModel(BaseModel):
    """Speech recognition model using Vosk"""

    def __init__(self, model_path: str, language: str = 'en', **kwargs):
        self.language = language
        super().__init__(model_path, **kwargs)

    def _initialize_model(self, **kwargs):
        """Initialize the Vosk model"""
        try:
            self._model = vosk.Model(self.model_path)
            self.logger.info(f"Initialized speech model for language: {self.language}")
        except Exception as e:
            self.logger.error(f"Error initializing speech model: {e}")
            raise

    def predict(self, audio_data: bytes) -> Optional[str]:
        """Recognize speech from audio data"""
        temp_file = "temp.wav"
        try:
            if not self.is_initialized():
                self.logger.error("Speech model not initialized")
                return None

            # Create a temporary WAV file
            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_data)

            # Process audio with Vosk
            with wave.open(temp_file, "rb") as wf:
                recognizer = vosk.KaldiRecognizer(self._model, wf.getframerate())
                recognizer.SetWords(True)

                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        if result.get("text"):
                            return result["text"]

                # Get final result
                result = json.loads(recognizer.FinalResult())
                return result.get("text")

        except Exception as e:
            self.logger.error(f"Error recognizing speech: {e}")
            return None
        finally:
            # Clean up temporary file
            try:
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.error(f"Error cleaning up temporary file: {e}")

    def detect_language(self, text: str) -> Optional[str]:
        """Detect language from text"""
        try:
            # Simple language detection based on character sets
            # This is a basic implementation - consider using a proper language detection library
            if any('\u0600' <= c <= '\u06FF' for c in text):  # Arabic
                return 'ar'
            elif any('\u4E00' <= c <= '\u9FFF' for c in text):  # Chinese
                return 'zh'
            elif any('\u3040' <= c <= '\u309F' for c in text):  # Japanese
                return 'ja'
            elif any('\uAC00' <= c <= '\uD7AF' for c in text):  # Korean
                return 'ko'
            elif any('\u0400' <= c <= '\u04FF' for c in text):  # Russian
                return 'ru'
            else:
                return 'en'  # Default to English
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")
            return None

    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages and their codes"""
        return {
            'English': 'en',
            'French': 'fr',
            'Arabic': 'ar',
            'Spanish': 'es',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Russian': 'ru',
            'Chinese': 'zh',
            'Japanese': 'ja',
            'Korean': 'ko'
        }

    def cleanup(self):
        """Clean up model resources"""
        try:
            self._model = None
            self.logger.info("Speech model cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up speech model: {e}")
