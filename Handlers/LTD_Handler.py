from transformers import MarianMTModel, MarianTokenizer, pipeline
from utils.Logging import Logger
from utils.Config import IO_CONFIG
import threading


class LTDHandler:
    def __init__(self):
        self.logger = Logger()
        self.logger.info("Initializing LTD Handler")
        self._translation_models = {}
        self._model_lock = threading.Lock()

    def load_language_model(self, language_code):
        # Process the request using the LD instance
        response = self.ld.process_request(language_code)
        return response

    def load_language_model_pair(self, source_lang, target_lang):
        """
        Get or load a translation model for the specified language pair.

        Args:
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            pipeline: A translation pipeline or None if failed
        """
        with self._model_lock:
            key = f"{source_lang}-{target_lang}"

            # Return existing model if available
            if key in self._translation_models:
                return self._translation_models[key]

            # Create new model
            try:
                model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
                translation_pipeline = pipeline("translation", model=model_name)
                self._translation_models[key] = translation_pipeline
                return translation_pipeline
            except Exception as e:
                self.Logger.error(f"Error loading translation model {source_lang}-{target_lang}: {e}")
                return None