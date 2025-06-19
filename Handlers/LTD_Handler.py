import os
from vosk import Model, KaldiRecognizer
from transformers import MarianMTModel, MarianTokenizer
from utils.Config import LTD_CONFIG
from utils.Logging import Logger
import threading


class LTDManager:
    def __init__(self):
        self.logger = Logger()
        self.logger.info("Initializing LTD Manager")

        # Model storage
        self._speech_models = {}  # {lang_code: vosk_model}
        self._translation_models = {}  # {src-dest: (model, tokenizer)}

        # Thread safety
        self._model_lock = threading.Lock()

        # Config
        self.supported_lang_codes = LTD_CONFIG['SUPPORTED_LANGUAGES']
        self.languages_map = {'arabic': 'ar', 'english': 'en', 'french': 'fr'}

    def load_speech_model(self, language):
        """
        Load Vosk speech recognition model for a specific language.

        Args:
            language (str): Language code or name

        Returns:
            Model: Loaded Vosk model or None if failed
        """
        with self._model_lock:
            try:
                # Convert language name to code if needed
                lang_code = self._get_language_code(language)
                if not lang_code:
                    raise ValueError(f"Unsupported language: {language}")

                # Return cached model if exists
                if lang_code in self._speech_models:
                    self.logger.info(f"Using cached speech model for {lang_code}")
                    return self._speech_models[lang_code]

                # Load new model
                model_path = os.path.join(LTD_CONFIG.get('VOSK_MODELS_DIR', ''), f"vosk-model-{lang_code}")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Speech model not found for {lang_code}")

                model = Model(model_path)
                self._speech_models[lang_code] = model
                self.logger.info(f"Loaded speech model for {lang_code}")
                return model

            except Exception as e:
                self.logger.error(f"Failed to load speech model: {str(e)}")
                return None

    def load_translation_model(self, source_lang, target_lang):
        """
        Load MarianMT translation model for a language pair.

        Args:
            source_lang (str): Source language code or name
            target_lang (str): Target language code or name

        Returns:
            tuple: (model, tokenizer) or None if failed
        """
        with self._model_lock:
            try:
                # Convert language names to codes if needed
                src_code = self._get_language_code(source_lang)
                tgt_code = self._get_language_code(target_lang)

                if not src_code or not tgt_code:
                    raise ValueError("Invalid language codes")

                model_key = f"{src_code}-{tgt_code}"

                # Return cached model if exists
                if model_key in self._translation_models:
                    self.logger.info(f"Using cached translation model for {model_key}")
                    return self._translation_models[model_key]

                # Load new model
                model_name = f'Helsinki-NLP/opus-mt-{src_code}-{tgt_code}'
                model = MarianMTModel.from_pretrained(model_name)
                tokenizer = MarianTokenizer.from_pretrained(model_name)

                self._translation_models[model_key] = (model, tokenizer)
                self.logger.info(f"Loaded translation model for {model_key}")
                return model, tokenizer

            except Exception as e:
                self.logger.error(f"Failed to load translation model: {str(e)}")
                return None

    def unload_model(self, model_type, language=None, source_lang=None, target_lang=None):
        """
        Unload specified model from RAM.

        Args:
            model_type (str): 'speech' or 'translation'
            language (str, optional): Language for speech model
            source_lang (str, optional): Source language for translation model
            target_lang (str, optional): Target language for translation model

        Returns:
            bool: True if successful, False otherwise
        """
        with self._model_lock:
            try:
                if model_type == 'speech' and language:
                    lang_code = self._get_language_code(language)
                    if lang_code in self._speech_models:
                        del self._speech_models[lang_code]
                        self.logger.info(f"Unloaded speech model for {lang_code}")
                        return True

                elif model_type == 'translation' and source_lang and target_lang:
                    src_code = self._get_language_code(source_lang)
                    tgt_code = self._get_language_code(target_lang)
                    model_key = f"{src_code}-{tgt_code}"

                    if model_key in self._translation_models:
                        del self._translation_models[model_key]
                        self.logger.info(f"Unloaded translation model for {model_key}")
                        return True

                return False

            except Exception as e:
                self.logger.error(f"Failed to unload model: {str(e)}")
                return False

    def _get_language_code(self, language):
        """Convert language name to code if needed."""
        if language in self.supported_lang_codes:
            return language
        return self.languages_map.get(language.lower())

    def cleanup(self):
        """Unload all models from RAM."""
        with self._model_lock:
            self._speech_models.clear()
            self._translation_models.clear()
            self.logger.info("All models unloaded")