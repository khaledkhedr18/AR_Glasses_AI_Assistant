from typing import Optional, Tuple, Dict, Any
import pytesseract
from .network import NetworkService
from .camera import CameraService
from core.models.base import ModelFactory
from config.settings import Settings
from core.utils.logging import Logger
from core.utils.progress import ProgressMonitor
from functools import lru_cache

class TranslationService:
    """Service for handling translation operations"""

    def __init__(self):
        self.logger = Logger()
        self.settings = Settings()
        self.progress = ProgressMonitor()
        self._models: Dict[str, Any] = {}
        self.network = NetworkService()
        self.camera = CameraService()
        self._initialize_models()

    def _initialize_models(self):
        """Initialize translation models"""
        try:
            # Initialize speech model
            speech_model_path = self.settings.get('models', 'speech', 'model_path')
            speech_language = self.settings.get('models', 'speech', 'language')
            self._models['speech'] = ModelFactory.get_model('speech', speech_model_path, language=speech_language)

            # Initialize translation model
            translation_model_path = self.settings.get('models', 'translation', 'model_path')
            source_lang = self.settings.get('models', 'translation', 'source_language')
            target_lang = self.settings.get('models', 'translation', 'target_language')
            self._models['translation'] = ModelFactory.get_model('translation', translation_model_path,
                                                               source_language=source_lang,
                                                               target_language=target_lang)

            self.logger.info("Translation models initialized successfully")

        except KeyError as e:
            self.logger.error(f"Missing required configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            raise

    @lru_cache(maxsize=1000)
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate text from source language to target language"""
        if not text or not text.strip():
            self.logger.warning("Empty text provided for translation")
            return None

        try:
            self.progress.update(0, "Starting translation")

            # Update model languages if needed
            if (self._models['translation'].source_language != source_lang or
                self._models['translation'].target_language != target_lang):
                self._models['translation'].set_languages(source_lang, target_lang)

            self.progress.update(25, "Translating text")
            translated_text = self._models['translation'].predict(text)

            if translated_text:
                self.progress.update(100, "Translation complete")
                return translated_text
            else:
                self.progress.update(0, "Translation failed")
                return None

        except Exception as e:
            self.logger.error(f"Error translating text: {e}")
            self.progress.update(0, f"Translation error: {str(e)}")
            return None

    def recognize_speech(self, audio_data: bytes) -> Optional[str]:
        """Recognize speech from audio data"""
        try:
            self.progress.update(0, "Starting speech recognition")

            self.progress.update(25, "Processing audio")
            text = self._models['speech'].recognize(audio_data)

            if text:
                self.progress.update(100, "Speech recognition complete")
                return text
            else:
                self.progress.update(0, "Speech recognition failed")
                return None

        except Exception as e:
            self.logger.error(f"Error recognizing speech: {e}")
            self.progress.update(0, f"Speech recognition error: {str(e)}")
            return None

    def clear_cache(self):
        """Clear translation cache"""
        self.translate_text.cache_clear()
        self.logger.info("Translation cache cleared")

    def translate_image(self, source_lang: str, target_lang: str) -> Tuple[bool, str]:
        """Translate text from an image"""
        if not self.camera.picam2:
            return False, "Camera not initialized"

        try:
            # Capture and process image
            frame = self.camera.capture_frame()
            if frame is None:
                return False, "Failed to capture image"

            # Extract text from image
            text = pytesseract.image_to_string(frame, lang=source_lang)
            if not text.strip():
                return False, "No text detected in image"

            # Translate the text
            translated_text = self.translate_text(text, source_lang, target_lang)
            return True, translated_text

        except Exception as e:
            return False, f"Error processing image: {str(e)}"

    def cleanup(self):
        """Cleanup service resources"""
        if self.network:
            self.network.cleanup()
        if self.camera:
            self.camera.cleanup()
        if self._models['translation']:
            self._models['translation'].cleanup()
        if self._models['speech']:
            self._models['speech'].cleanup()
