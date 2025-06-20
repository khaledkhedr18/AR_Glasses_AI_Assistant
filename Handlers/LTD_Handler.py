import os
import gc
import threading
import torch
from vosk import Model
from transformers import MarianMTModel, MarianTokenizer
from utils.Services import Services
from utils.Config import LTD_CONFIG, SERVICES_CONFIG
from utils.Logging import Logger
from concurrent.futures import ThreadPoolExecutor, as_completed


class LTDHandler:
    # Class-level variables for model loading state
    _models_loaded = False
    _speech_models = {}
    _translation_models = {}
    _model_lock = threading.Lock()
    _loading_complete = threading.Event()

    def __init__(self):
        """
        Initialize LTD Handler and trigger parallel loading of all models.

        Sets up:
        - Logger for tracking operations
        - Thread-safe model storage
        - Language mappings and configurations
        - Model loading completion event
        """
        self.logger = Logger()
        self.logger.info("Initializing LTD Handler")
        self.service = Services()

        # Language configuration
        self.languages_map = SERVICES_CONFIG['LANGUAGES_MAP']

        # Load models only once across all instances
        with LTDHandler._model_lock:
            if not LTDHandler._models_loaded:
                self.__preload_all_models()
                LTDHandler._models_loaded = True

    def get_translation_model(self, source_lang, target_lang):
        """
        Retrieve a preloaded translation model.

        Args:
            source_lang (str): Source language name or code
            target_lang (str): Target language name or code

        Returns:
            tuple: (model, tokenizer, bool) - model, tokenizer and success flag
        """
        if not self._loading_complete.wait(timeout=LTD_CONFIG['MODELS_LOAD_TIMEOUT']):  # Add timeout
            self.logger.error("Timeout waiting for models to load")
            return None, None, False

        # Wait for all models to load
        src_code = self.service.get_language_code(source_lang)
        tgt_code = self.service.get_language_code(target_lang)
        if not src_code or not tgt_code:
            self.logger.error(f"Invalid language codes: {source_lang}, {target_lang}")
            return None, None, False

        model_key = f"{src_code}-{tgt_code}"
        model_pair = self._translation_models.get(model_key)

        return (*model_pair, True) if model_pair else (None, None, False)

    def get_speech_model(self, language):
        """
        Retrieve a preloaded speech recognition model.

        Args:
            language (str): Language name or code

        Returns:
            tuple: (Model, bool) - Vosk model and success flag
        """
        if not self._loading_complete.wait(timeout=LTD_CONFIG['MODELS_LOAD_TIMEOUT']):  # Add timeout
            self.logger.error("Timeout waiting for models to load")
            return None, False

        lang_code = self.service.get_language_code(language)
        if not lang_code:
            self.logger.error(f"Invalid language code: {language}")
            return None, False

        model = self._speech_models.get(lang_code)
        return (model, True) if model else (None, False)

    def cleanup(self):
        """Clean up all loaded models and free memory."""
        with self._model_lock:
            try:
                # Clean up speech models
                for model in self._speech_models.values():
                    if hasattr(model, '__del__'):
                        model.__del__()
                self._speech_models.clear()

                # Clean up translation models - updated to avoid unused variables
                for model_pair in self._translation_models.values():
                    del model_pair[0]  # Delete model
                    del model_pair[1]  # Delete tokenizer
                self._translation_models.clear()

                # Reset class state
                LTDHandler._models_loaded = False
                self._loading_complete.clear()
                gc.collect()  # Force garbage collection

                self.logger.info("All models unloaded and resources cleaned up")
            except Exception as e:
                self.logger.error(f"Failed to cleanup: {str(e)}")
                raise

    def __preload_all_models(self):
        """Preload all speech and translation models using thread pool."""
        self.logger.info("Starting parallel model loading...")

        vosk_base_path = LTD_CONFIG['VOSK_MODELS_DIR']
        languages = list(self.languages_map.values())
        loading_tasks = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            for lang in languages:
                speech_model_path = os.path.join(vosk_base_path, f'vosk-model-{lang}')
                if os.path.exists(speech_model_path):
                    loading_tasks.append(
                        executor.submit(self._load_speech_model_threaded, lang)
                    )

            for src_lang in languages:
                for tgt_lang in languages:
                    if src_lang != tgt_lang:
                        loading_tasks.append(
                            executor.submit(
                                self._load_translation_model_threaded,
                                src_lang,
                                tgt_lang
                            )
                        )

            for future in as_completed(loading_tasks):
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Model loading error: {str(e)}")

        self._loading_complete.set()
        self.logger.info("All models loaded successfully")

    def _load_speech_model_threaded(self, language):
        """
        Thread-safe method to load a speech recognition model.

        Args:
            language (str): Language code for the model to load

        Loads Vosk model for the specified language and stores it in _speech_models.
        Uses thread lock to ensure thread-safe model storage.
        """
        try:
            model_path = os.path.join(LTD_CONFIG['VOSK_MODELS_DIR'], f'vosk-model-{language}')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Speech model not found for {language}")

            model = Model(model_path)
            with self._model_lock:
                self._speech_models[language] = model
                self.logger.info(f"Loaded speech model for {language}")

        except Exception as e:
            self.logger.error(f"Failed to load speech model for {language}: {str(e)}")

    def _load_translation_model_threaded(self, src_lang, tgt_lang):
        """
        Thread-safe method to load a translation model.

        Args:
            src_lang (str): Source language code
            tgt_lang (str): Target language code

        Loads MarianMT model and tokenizer for the language pair and stores them.
        Uses thread lock to ensure thread-safe model storage.
        """
        try:
            model_key = f"{src_lang}-{tgt_lang}"
            model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'

            model = MarianMTModel.from_pretrained(model_name, torch_dtype=torch.float32)
            tokenizer = MarianTokenizer.from_pretrained(model_name)

            with self._model_lock:
                self._translation_models[model_key] = (model, tokenizer)
                self.logger.info(f"Loaded translation model for {model_key}")

        except Exception as e:
            self.logger.error(f"Failed to load translation model for {model_key}: {str(e)}")