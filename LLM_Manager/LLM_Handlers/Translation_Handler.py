from transformers import MarianMTModel, MarianTokenizer, pipeline
import threading
from utils.logging import Logger

class Translation_Handler:
    """
    Handles all translation operations without direct dependencies on other handlers.
    """

    def __init__(self):
        """Initialize the Translation_Handler."""
        self.logger = Logger()
        self.logger.info("Initializing Translation_Handler")

        # Model caching
        self._translation_models = {}
        self._model_lock = threading.Lock()

    def load_multilingual_to_english_model(self):
        """
        Initialize and return a model for multilingual to English translation.

        Returns:
            tuple: A tuple containing the model and tokenizer instances
        """
        model_name = "Helsinki-NLP/opus-mt-mul-en"

        with self._model_lock:
            # Check if we already have this model loaded
            if model_name in self._translation_models:
                return self._translation_models[model_name]

            # Load the model
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                self._translation_models[model_name] = (model, tokenizer)
                return model, tokenizer
            except Exception as e:
                self.logger.error(f"Error loading multilingual model: {e}")
                return None, None

    def translate_text_to_english(self, text):
        """
        Translate the given text into English.

        Args:
            text (str): The text to translate

        Returns:
            str: Translated text or None if failed
        """
        model, tokenizer = self.load_multilingual_to_english_model()

        if not model or not tokenizer:
            return None

        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = model.generate(**inputs)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            return None

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
                self.logger.error(f"Error loading translation model {source_lang}-{target_lang}: {e}")
                return None

    def translate_with_best_available_method(self, text, source_lang, target_lang):
        """
        Translate text using the best available method.

        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            str: Translated text or None if failed
        """
        if not text or not text.strip():
            return None

        # For this implementation we'll use the offline translation since you mentioned
        # the LLM manager will only be used in offline mode
        return self.perform_offline_translation(text, source_lang, target_lang)

    def perform_offline_translation(self, text, source_lang, target_lang):
        """
        Translate text using offline models.

        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            str: Translated text or None if failed
        """
        try:
            pipe = self.load_language_model_pair(source_lang, target_lang)
            if not pipe:
                return None

            translated_text = pipe(text)[0]['translation_text']
            return translated_text

        except Exception as e:
            self.logger.error(f"Offline translation error: {e}")
            return None

    def detect_translation_exit_phrase(self, text):
        """
        Check if the text contains phrases indicating a desire to exit translation mode.

        Args:
            text (str): Text to check

        Returns:
            bool: True if exit phrase detected
        """
        # Check in original text
        stop_phrases = {"stop", "exit", "quit", "end", "get out", "goodbye"}
        if any(phrase in text.lower() for phrase in stop_phrases):
            return True

        # Try to translate to English for checking
        english_text = self.translate_text_to_english(text)
        if english_text:
            english_text = english_text.lower()
            return any(phrase in english_text for phrase in stop_phrases)

        return False

    def release_translation_resources(self):
        """Clean up resources used by the translation handler."""
        self.logger.info("Releasing translation resources")

        # Clear model cache
        with self._model_lock:
            for model_key in list(self._translation_models.keys()):
                self._translation_models[model_key] = None

        self._translation_models.clear()
