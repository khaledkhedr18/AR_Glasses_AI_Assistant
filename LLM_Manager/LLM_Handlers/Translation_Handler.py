from transformers import MarianMTModel, MarianTokenizer, pipeline
import threading
from PyQt5.QtCore import QObject, QTimer
from IO_Manager.IO_Handlers.Audio_Handler import AudioHandler
from Network_Manager.Network_Handler import Network_Handler

class Translation_Handler(QObject):
    """
    Handles all translation operations, focusing only on translation functionality.

    This class provides methods for translating text using local models,
    with support for both online and offline translation.
    """

    def __init__(self, audio_handler=None, network_handler=None):
        """
        Initialize the Translation_Handler.

        Args:
            audio_handler (AudioHandler, optional): An instance of AudioHandler for speech feedback.
                If None, a new instance will be created.
            network_handler (Network_Handler, optional): An instance of Network_Handler for online translations.
                If None, a new instance will be created.
        """
        super().__init__()
        self.audio_handler = audio_handler or AudioHandler()
        self.network_handler = network_handler or Network_Handler()

        # Model caching
        self._translation_models = {}
        self._model_lock = threading.Lock()

    def translate_en(self):
        """
        Initializes and returns a MarianMTModel and a MarianTokenizer
        for multilingual to English translation.

        Returns:
            tuple: A tuple containing the MarianMTModel and MarianTokenizer instances.
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
                print(f"Error loading multilingual model: {e}")
                self.audio_handler.speak("Failed to load translation model")
                return None, None

    def translate_to_english(self, text):
        """
        Translates the given text into English.

        Args:
            text (str): The text to be translated.

        Returns:
            str: The translated text in English, or None if translation fails.
        """
        model, tokenizer = self.translate_en()

        if not model or not tokenizer:
            return None

        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            translated = model.generate(**inputs)
            return tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Translation error: {e}")
            return None

    def get_translation_model(self, source_lang, target_lang):
        """
        Get or load a translation model for the specified language pair.

        Args:
            source_lang (str): Source language code (e.g., "en", "ar")
            target_lang (str): Target language code (e.g., "en", "ar")

        Returns:
            pipeline: A translation pipeline, or None if loading fails
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
                print(f"Error loading translation model {source_lang}-{target_lang}: {e}")
                self.audio_handler.speak("Failed to load translation model")
                return None

    def translate_text(self, text, source_lang, target_lang):
        """
        Translate text using online or offline methods based on connectivity.

        Args:
            text (str): The text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            str or None: Translated text if successful, None otherwise
        """
        if not text or not text.strip():
            return None

        # Try online translation first if available
        if self.network_handler.check_internet_connection():
            translated = self.translate_online(text, source_lang, target_lang)
            if translated:
                return translated

        # Fall back to offline translation
        return self.translate_offline(text, source_lang, target_lang)

    def translate_offline(self, text, source_lang, target_lang):
        """
        Translate text using offline models.

        Args:
            text (str): The text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            str or None: Translated text if successful, None otherwise
        """
        try:
            pipe = self.get_translation_model(source_lang, target_lang)
            if not pipe:
                return None

            translated_text = pipe(text)[0]['translation_text']
            return translated_text

        except Exception as e:
            print(f"Offline translation error: {e}")
            return None

    def translate_online(self, text, source_lang, target_lang):
        """
        Translate text using online translation service via NetworkHandler.

        Args:
            text (str): The text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            str or None: Translated text if successful, None otherwise
        """
        try:
            response = self.network_handler.send_and_receive("text", source_lang, target_lang, text=text)
            return response

        except Exception as e:
            print(f"Online translation error: {e}")
            return None

    def should_stop_translation(self, text):
        """
        Check if the given text contains stop phrases in any language.

        Args:
            text (str): The text to check

        Returns:
            bool: True if a stop phrase is detected, False otherwise
        """
        # Check in original text
        stop_phrases = {"stop", "exit", "quit", "end", "get out", "goodbye"}
        if any(phrase in text.lower() for phrase in stop_phrases):
            return True

        # Try to translate to English for checking
        english_text = self.translate_to_english(text)
        if english_text:
            english_text = english_text.lower()
            return any(phrase in english_text for phrase in stop_phrases)

        return False

    def handle_online_translation(self, text, signals, source_lang, target_lang):
        """
        Handle online translation flow with UI updates.

        Args:
            text (str): The text to translate
            signals: Communication signals for UI updates
            source_lang (str): Source language code
            target_lang (str): Target language code
        """
        if hasattr(signals, 'update_ai_speech'):
            signals.update_ai_speech.emit("Processing online translation...")

        response = self.network_handler.send_and_receive("text", source_lang, target_lang, text=text)

        if response:
            if hasattr(signals, 'update_ai_speech'):
                signals.update_ai_speech.emit(response)
                QTimer.singleShot(2000, lambda: signals.update_ai_speech.emit("Listening"))

            self.audio_handler.speak(response)
            return True
        else:
            if hasattr(signals, 'update_ai_speech'):
                signals.update_ai_speech.emit("Translation service unavailable")

            self.audio_handler.speak("Online translation failed")
            print("Online translation failed")
            return False

    def handle_offline_translation(self, text, signals, source_lang, target_lang):
        """
        Handle offline translation flow with UI updates.

        Args:
            text (str): The text to translate
            signals: Communication signals for UI updates
            source_lang (str): Source language code
            target_lang (str): Target language code
        """
        if hasattr(signals, 'update_ai_speech'):
            signals.update_ai_speech.emit("Processing offline translation...")

        pipeline = self.get_translation_model(source_lang, target_lang)
        if not pipeline:
            if hasattr(signals, 'update_ai_speech'):
                signals.update_ai_speech.emit("Offline model unavailable")

            self.audio_handler.speak("Translation resources missing")
            print("Translation resources missing")
            return False

        try:
            translated_text = pipeline(text)[0]['translation_text']

            if translated_text:
                if hasattr(signals, 'update_ai_speech'):
                    signals.update_ai_speech.emit(translated_text)
                    QTimer.singleShot(2000, lambda: signals.update_ai_speech.emit("Listening"))

                if hasattr(signals, 'update_output'):
                    signals.update_output.emit(f"Translated: {translated_text}")

                self.audio_handler.speak(translated_text)
                print(translated_text)
                return True
        except Exception as e:
            print(f"Translation error: {e}")

        if hasattr(signals, 'update_ai_speech'):
            signals.update_ai_speech.emit("Offline translation failed")

        self.audio_handler.speak("Could not translate text")
        print("Could not translate text")
        return False

    def cleanup(self):
        """
        Clean up resources used by the translation handler.
        """
        # Clear model cache
        with self._model_lock:
            for model_key in list(self._translation_models.keys()):
                self._translation_models[model_key] = None

        self._translation_models.clear()
