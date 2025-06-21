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

    Usage:
    handler1 = LTDHandler()  # First instance - will load models
    handler2 = LTDHandler()  # Second instance - will use already loaded models
    """

    # Singleton instance and state management
    _instance = None
    _instance_lock = threading.Lock()

    # Model management
    _models_initialized = False
    _initialization_lock = threading.Lock()
    _loading_complete = threading.Event()

    # Global model storage
    _speech_models = {}
    _translation_models = {}

    def __new__(cls):
        """Ensure single instance creation (thread-safe)"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super(LTDHandler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize handler with memory-optimized settings (thread-safe)"""
        # Skip full initialization if already done
        if self._models_initialized:
            return

        # Ensure only one thread performs initialization
        with self._initialization_lock:
            if not self._models_initialized:
                self.logger = Logger()
                self.logger.info("Initializing LTD Handler")

                # Core components initialization
                self.service = Services()
                self.models_dir = LTDConfig.TRANSLATION['TRANSLATION_MODELS_DIR']
                self.recognizer_model_path = ServicesConfig.RECOGNITION['RECOGNITION_MODEL_DIR']
                self.model_timeout = LTDConfig.TRANSLATION['MODELS_LOAD_TIMEOUT']

                # Load models only if not already loaded
                self.__load_models()

                # Mark initialization as complete
                self._models_initialized = True
                self.logger.info("LTD Handler initialization completed")

    def __load_models(self):
        """Initialize and load models if not already loaded (thread-safe)"""
        if self._loading_complete.is_set():
            self.logger.info("Models already loaded, using existing models")
            return

        with self._initialization_lock:
            if not self._loading_complete.is_set():
                try:
                    # Clear memory before loading
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

                    self.__ensure_model_directories()
                    self.__preload_all_models()

                    # Mark loading as complete only if successful
                    self._loading_complete.set()
                    self.logger.info("Initial model loading completed successfully")
                except Exception as e:
                    self.logger.error(f"Model loading failed: {str(e)}")
                    # Reset state on failure
                    self._loading_complete.clear()
                    raise

    def __ensure_model_directories(self):
        """Create and verify required model directories (thread-safe)"""
        directories = [
            self.models_dir,
            self.recognizer_model_path,
            LTDConfig.TRANSLATION['MODELS_DIR']
        ]
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                if not os.access(directory, os.W_OK):
                    raise PermissionError(f"Cannot write to {directory}")
            except Exception as e:
                self.logger.error(f"Directory creation failed: {str(e)}")
                raise

    def __preload_all_models(self):
        """Load models using thread pool with resource management (thread-safe)"""
        self.logger.info("Starting model loading process")
        loading_tasks = []

        try:
            with ThreadPoolExecutor(max_workers=LTDConfig.TRANSLATION['MAX_WORKERS']) as executor:
                # Load speech model first (smaller memory footprint)
                speech_model_path = os.path.join(
                    ServicesConfig.RECOGNITION['VOSK_MODEL_PATH'],
                    ServicesConfig.RECOGNITION['VOSK_MODELS']['en']
                )
                loading_tasks.append(
                    executor.submit(self.__load_speech_model, 'en', speech_model_path)
                )

                # Load translation models sequentially
                for src_lang, tgt_lang in LTDConfig.TRANSLATION['SUPPORTED_TRANSLATION_PAIRS']:
                    loading_tasks.append(
                        executor.submit(
                            self.__load_translation_model,
                            src_lang,
                            tgt_lang
                        )
                    )
                    # Force garbage collection between models
                    gc.collect()

                # Wait for all tasks to complete
                for future in as_completed(loading_tasks):
                    try:
                        future.result()
                    except Exception as e:
                        self.logger.error(f"Model loading task failed: {str(e)}")
                        raise

        except Exception as e:
            self.logger.error(f"Model loading process failed: {str(e)}")
            raise

    def __load_speech_model(self, language, model_path):
        """Load speech model if not already loaded (thread-safe)"""
        try:
            # Check if model is already loaded
            if language in self._speech_models:
                self.logger.info(f"Speech model already loaded for: {language}")
                return

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Speech model not found: {model_path}")

            model = Model(model_path)
            with self._initialization_lock:
                self._speech_models[language] = model
                self.logger.info(f"Speech model loaded: {language}")

        except Exception as e:
            self.logger.error(f"Speech model loading failed: {str(e)}")
            raise

    def __load_translation_model(self, src_lang, tgt_lang):
        """Load translation model if not already loaded (thread-safe)"""
        try:
            model_key = f"{src_lang}-{tgt_lang}"

            # Check if model is already loaded
            if model_key in self._translation_models:
                self.logger.info(f"Translation model already loaded for: {model_key}")
                return

            # Get specific model name for language pair
            model_name = LTDConfig.TRANSLATION['MODEL_NAMES'].get(model_key)
            if not model_name:
                raise ValueError(f"No model name found for language pair: {model_key}")

            # Load model with memory optimizations
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

            with self._initialization_lock:
                self._translation_models[model_key] = (model, tokenizer)
                self.logger.info(f"Translation model loaded: {model_key}")

        except Exception as e:
            self.logger.error(f"Translation model loading failed: {str(e)}")
            raise

    def get_translation_model(self, source_lang, target_lang):
        """Get translation model from global cache (thread-safe)"""
        # Wait for models to be loaded
        if not self._loading_complete.wait(timeout=self.model_timeout):
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
        model_pair = self._translation_models.get(model_key)

        return (*model_pair, True) if model_pair else (None, None, False)

    def get_speech_model(self, language):
        """Get speech model from global cache (thread-safe)"""
        # Wait for models to be loaded
        if not self._loading_complete.wait(timeout=self.model_timeout):
            self.logger.error("Model loading timeout exceeded")
            return None, False

        lang_code = self.service.get_language_code(language)
        if not lang_code or lang_code not in LTDConfig.TRANSLATION['SUPPORTED_SPEECH_MODELS']:
            self.logger.error(f"Unsupported speech language: {language}")
            return None, False

        model = self._speech_models.get(lang_code)
        return (model, True) if model else (None, False)

    def cleanup(self):
        """Clean up global resources (thread-safe)"""
        with self._initialization_lock:
            try:
                # Clean speech models
                for model in list(self._speech_models.values()):
                    if hasattr(model, '__del__'):
                        model.__del__()
                self._speech_models.clear()

                # Clean translation models
                for model_pair in list(self._translation_models.values()):
                    if model_pair:
                        del model_pair[0]  # Delete model
                        del model_pair[1]  # Delete tokenizer
                self._translation_models.clear()

                # Reset all state flags
                LTDHandler._models_initialized = False
                self._loading_complete.clear()

                # Aggressive memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                self.logger.info("All models unloaded and memory cleaned")
            except Exception as e:
                self.logger.error(f"Cleanup failed: {str(e)}")
                raise