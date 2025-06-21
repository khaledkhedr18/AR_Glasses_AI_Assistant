import os
import gc
import torch
from vosk import Model
from transformers import MarianMTModel, MarianTokenizer
from utils.Services import Services
from utils.Config import LTDConfig, ServicesConfig
from utils.Logging import Logger
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


class LTDHandler:
    """
    Language Translation and Detection Handler optimized for Raspberry Pi 4 8GB RAM.
    Implements singleton pattern to prevent multiple model loading.

    Memory Management:
    - Speech Model (Vosk): ~50MB
    - Translation Models (MarianMT): ~300MB each
    - Total Memory Usage: ~950MB base + ~200MB overhead
    """

    # Class-level variables for singleton pattern
    _instance = None
    _initialized = False
    _global_lock = threading.Lock()
    _global_speech_models = {}
    _global_translation_models = {}
    _global_loading_complete = threading.Event()

    def __new__(cls):
        """Implement singleton pattern"""
        with cls._global_lock:
            if cls._instance is None:
                cls._instance = super(LTDHandler, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        """Initialize handler with memory-optimized settings"""
        # Skip initialization if already done
        if LTDHandler._initialized:
            return

        with self._global_lock:
            if not LTDHandler._initialized:
                self.logger = Logger()
                self.logger.info("Initializing LTD Handler")

                # Core components initialization
                self.service = Services()
                self.models_dir = LTDConfig.TRANSLATION['TRANSLATION_MODELS_DIR']
                self.recognizer_model_path = ServicesConfig.RECOGNITION['RECOGNITION_MODEL_DIR']
                self.model_timeout = LTDConfig.TRANSLATION['MODELS_LOAD_TIMEOUT']

                # Initialize models if not already loaded
                self.__load_models()
                LTDHandler._initialized = True

    def __load_models(self):
        """Initialize and load models if not already loaded"""
        if self._global_loading_complete.is_set():
            self.logger.info("Models already loaded, skipping initialization")
            return

        with self._global_lock:
            if not self._global_loading_complete.is_set():
                # Clear memory before loading
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                self.__ensure_model_directories()
                self.__preload_all_models()

    def __ensure_model_directories(self):
        """Create and verify required model directories"""
        directories = [
            self.models_dir,
            self.recognizer_model_path,
            LTDConfig.TRANSLATION['MODELS_DIR']
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            if not os.access(directory, os.W_OK):
                self.logger.error(f"Directory not writable: {directory}")
                raise PermissionError(f"Cannot write to {directory}")

    def __preload_all_models(self):
        """Load models using thread pool with resource management"""
        self.logger.info("Starting optimized model loading process")
        loading_tasks = []

        try:
            with ThreadPoolExecutor(max_workers=LTDConfig.TRANSLATION['MAX_WORKERS']) as executor:
                # Load speech model first
                speech_model_path = os.path.join(
                    ServicesConfig.RECOGNITION['VOSK_MODEL_PATH'],
                    ServicesConfig.RECOGNITION['VOSK_MODELS']['en']
                )
                loading_tasks.append(
                    executor.submit(self.__load_speech_model, 'en', speech_model_path)
                )

                # Load translation models
                for src_lang, tgt_lang in LTDConfig.TRANSLATION['SUPPORTED_TRANSLATION_PAIRS']:
                    loading_tasks.append(
                        executor.submit(
                            self.__load_translation_model,
                            src_lang,
                            tgt_lang
                        )
                    )
                    gc.collect()

                # Monitor completion
                for future in as_completed(loading_tasks):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Model loading task failed: {str(e)}")
                        raise

            self._global_loading_complete.set()
            self.logger.info("Models loaded and cached globally")

        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            self._global_loading_complete.set()
            raise

    def __load_speech_model(self, language, model_path):
        """Load speech model if not already loaded"""
        try:
            if language in self._global_speech_models:
                self.logger.info(f"Speech model already loaded for: {language}")
                return

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Speech model not found: {model_path}")

            model = Model(model_path)
            with self._global_lock:
                self._global_speech_models[language] = model
                self.logger.info(f"Speech model loaded: {language}")

        except Exception as e:
            self.logger.error(f"Speech model loading failed: {str(e)}")
            raise

    def __load_translation_model(self, src_lang, tgt_lang):
        """Load translation model if not already loaded"""
        try:
            model_key = f"{src_lang}-{tgt_lang}"

            if model_key in self._global_translation_models:
                self.logger.info(f"Translation model already loaded for: {model_key}")
                return

            model_name = LTDConfig.TRANSLATION['MODEL_NAMES'].get(model_key)
            if not model_name:
                raise ValueError(f"No model name found for language pair: {model_key}")

            model = MarianMTModel.from_pretrained(
                model_name,
                torch_dtype=getattr(torch, LTDConfig.TRANSLATION['TORCH_DTYPE']),
                cache_dir=LTDConfig.TRANSLATION['MODELS_DIR'],
                low_cpu_mem_usage=True,
                return_dict=False
            )

            tokenizer = MarianTokenizer.from_pretrained(
                model_name,
                cache_dir=LTDConfig.TRANSLATION['MODELS_DIR'],
                model_max_length=512
            )

            with self._global_lock:
                self._global_translation_models[model_key] = (model, tokenizer)
                self.logger.info(f"Translation model loaded: {model_key}")

        except Exception as e:
            self.logger.error(f"Translation model loading failed: {str(e)}")
            raise

    def get_translation_model(self, source_lang, target_lang):
        """Get translation model from global cache"""
        if not self._global_loading_complete.wait(timeout=self.model_timeout):
            self.logger.error("Model loading timeout exceeded")
            return None, None, False

        src_code = self.service.get_language_code(source_lang)
        tgt_code = self.service.get_language_code(target_lang)

        if not src_code or not tgt_code:
            self.logger.error(f"Invalid language codes: {source_lang}, {target_lang}")
            return None, None, False

        pair = (src_code, tgt_code)
        if pair not in LTDConfig.TRANSLATION['SUPPORTED_TRANSLATION_PAIRS']:
            self.logger.error(f"Unsupported translation pair: {pair}")
            return None, None, False

        model_key = f"{src_code}-{tgt_code}"
        model_pair = self._global_translation_models.get(model_key)

        return (*model_pair, True) if model_pair else (None, None, False)

    def get_speech_model(self, language):
        """Get speech model from global cache"""
        if not self._global_loading_complete.wait(timeout=self.model_timeout):
            self.logger.error("Model loading timeout exceeded")
            return None, False

        lang_code = self.service.get_language_code(language)
        if not lang_code or lang_code not in LTDConfig.TRANSLATION['SUPPORTED_SPEECH_MODELS']:
            self.logger.error(f"Unsupported speech language: {language}")
            return None, False

        model = self._global_speech_models.get(lang_code)
        return (model, True) if model else (None, False)

    def cleanup(self):
        """Clean up global resources"""
        with self._global_lock:
            try:
                # Clean speech models
                for model in list(self._global_speech_models.values()):
                    if hasattr(model, '__del__'):
                        model.__del__()
                self._global_speech_models.clear()

                # Clean translation models
                for model_pair in list(self._global_translation_models.values()):
                    if model_pair:
                        del model_pair[0]  # Delete model
                        del model_pair[1]  # Delete tokenizer
                self._global_translation_models.clear()

                # Reset singleton state
                LTDHandler._initialized = False
                self._global_loading_complete.clear()

                # Aggressive cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.logger.info("Global models unloaded and memory cleaned")
            except Exception as e:
                self.logger.error(f"Cleanup failed: {str(e)}")
                raise