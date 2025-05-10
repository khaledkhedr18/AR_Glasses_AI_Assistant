from typing import Optional, Dict, Any
from transformers import MarianMTModel, MarianTokenizer
from .base import BaseModel
from core.utils.logging import Logger

class TranslationModel(BaseModel):
    """Translation model using MarianMT"""

    def __init__(self, model_path: str, source_language: str = 'en', target_language: str = 'fr', **kwargs):
        self.source_language = source_language
        self.target_language = target_language
        super().__init__(model_path, **kwargs)

    def _initialize_model(self, **kwargs):
        """Initialize the MarianMT model"""
        try:
            self._model = MarianMTModel.from_pretrained(self.model_path)
            self._tokenizer = MarianTokenizer.from_pretrained(self.model_path)
            self.logger.info(f"Initialized translation model for {self.source_language} to {self.target_language}")
        except Exception as e:
            self.logger.error(f"Error initializing translation model: {e}")
            raise

    def predict(self, text: str) -> Optional[str]:
        """Translate text from source language to target language"""
        try:
            if not self.is_initialized():
                self.logger.error("Translation model not initialized")
                return None

            # Tokenize input text
            inputs = self._tokenizer(text, return_tensors="pt", padding=True)

            # Generate translation
            translated = self._model.generate(**inputs)

            # Decode translation
            translated_text = self._tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

            return translated_text

        except Exception as e:
            self.logger.error(f"Error translating text: {e}")
            return None

    def set_languages(self, source_lang: str, target_lang: str):
        """Set source and target languages"""
        try:
            if source_lang != self.source_language or target_lang != self.target_language:
                self.source_language = source_lang
                self.target_language = target_lang

                # Reload model for new language pair
                model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
                self._model = MarianMTModel.from_pretrained(model_name)
                self._tokenizer = MarianTokenizer.from_pretrained(model_name)

                self.logger.info(f"Updated translation model for {source_lang} to {target_lang}")
        except Exception as e:
            self.logger.error(f"Error setting languages: {e}")
            raise

    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """Get supported language pairs"""
        return {
            'English': {
                'French': 'en-fr',
                'Arabic': 'en-ar',
                'Spanish': 'en-es',
                'German': 'en-de',
                'Italian': 'en-it',
                'Portuguese': 'en-pt',
                'Russian': 'en-ru',
                'Chinese': 'en-zh',
                'Japanese': 'en-ja',
                'Korean': 'en-ko'
            },
            'French': {
                'English': 'fr-en',
                'Arabic': 'fr-ar',
                'Spanish': 'fr-es',
                'German': 'fr-de',
                'Italian': 'fr-it'
            },
            'Arabic': {
                'English': 'ar-en',
                'French': 'ar-fr'
            }
        }

    def cleanup(self):
        """Clean up model resources"""
        try:
            self._model = None
            self._tokenizer = None
            self.logger.info("Translation model cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up translation model: {e}")
