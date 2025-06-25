import gc
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from transformers import pipeline
from vosk import Model

from Handlers.OCR_Handler import OCRHandler
from Handlers.Translation_Handler import TranslationHandler
from utils.Config import ML_CONFIG, SERVICES_CONFIG, LLM_CONFIG
from utils.Logging import Logger
from utils.Services import Services


class LLMManager:
    """
    Central manager for all language model operations.
    Handles translation and extraction of text from images and speech.
    """

    class __ModelLoader:
        """Private model loader handler for managing ML models.
        Allows multiple instances but loads models only once."""

        # Class-level variables for shared resources
        __models_load_lock = threading.Lock()  # Lock for model loading
        __models_loaded = False  # Flag to track if models are loaded

        # Shared model storage across all instances
        __speech_models = {}  # Stores speech recognition models
        __translation_pipelines = {}  # Stores translation pipelines

        # Class-level event for tracking loading completion
        __loading_complete = threading.Event()

        def __init__(self):
            """Initialize handler instance with shared resources"""
            # Initialize instance-specific attributes
            self.service = Services()
            self.logger = Logger()
            self.logger.info("Initializing Model Loader Handler")

            # Core configuration initialization
            self.__load_config()

            # Load models only for first instance
            with self.__models_load_lock:
                if not self.__models_loaded:
                    try:
                        self.__load_models()
                        self.__class__.__models_loaded = True
                        self.__loading_complete.set()
                        self.logger.info("Initial model loading completed successfully")
                    except Exception as e:
                        self.logger.error(f"Model loading failed: {str(e)}")
                        self.__loading_complete.clear()
                        raise

        def __load_config(self):
            """Load configuration settings for models"""
            translation_config = ML_CONFIG.get('TRANSLATION', {})
            recognition_config = SERVICES_CONFIG.get('RECOGNITION', {})

            # Initialize configuration attributes with default values and expand paths
            self.models_cache_dir = os.path.expanduser(translation_config.get('MODELS_CACHE_DIR', './models/cache'))
            self.models_dir = os.path.expanduser(translation_config.get('TRANSLATION_MODELS_DIR', './models/translation'))
            self.models_load_timeout = translation_config.get('MODELS_LOAD_TIMEOUT', 30)
            self.supported_translation_pairs = translation_config.get('SUPPORTED_TRANSLATION_PAIRS', [])
            self.supported_speech_models = translation_config.get('SUPPORTED_SPEECH_MODELS', [])

            # Expand the tilde in the path
            self.recognizer_model_path = os.path.expanduser(recognition_config.get('VOSK_MODEL_DIR', './models/vosk'))

        def get_translation_model(self, source_lang: str, target_lang: str) -> tuple:
            """Get translation pipeline from shared storage

            Args:
                source_lang: Source language code
                target_lang: Target language code

            Returns:
                tuple: (pipeline, tokenizer, success_flag)
            """
            if not self.__loading_complete.wait(timeout=self.models_load_timeout):
                self.logger.error("Model loading timeout exceeded")
                return None, None, False

            model_key = f"{source_lang}-{target_lang}"
            translation_pipeline = self.__translation_pipelines.get(model_key)

            return (translation_pipeline, None, True) if translation_pipeline else (None, None, False)

        def get_speech_model(self, language: str) -> tuple:
            """Get speech model from shared storage

            Args:
                language: Language code

            Returns:
                tuple: (model, success_flag)
            """
            if not self.__loading_complete.wait(timeout=self.models_load_timeout):
                self.logger.error("Model loading timeout exceeded")
                return None, False

            model = self.__speech_models.get(language)
            return (model, True) if model else (None, False)

        def __load_models(self):
            """Initialize and load models into shared storage"""
            try:
                gc.collect()  # Clean memory before loading
                self.__ensure_model_directories()
                self.__preload_all_models()

            except Exception as e:
                self.logger.error(f"Model loading failed: {str(e)}")
                raise

        def __ensure_model_directories(self):
            """Create and verify required model directories"""
            directories = [
                self.recognizer_model_path,
                self.models_dir,
                self.models_cache_dir
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
            """Load models using thread pool with resource management"""
            self.logger.info("Starting model loading process")
            loading_tasks = []

            try:
                with ThreadPoolExecutor(max_workers=ML_CONFIG.get('MAX_WORKERS', 2)) as executor:
                    # Load speech recognition model
                    speech_model_path = os.path.join(
                        SERVICES_CONFIG.get('RECOGNITION', {}).get('VOSK_MODEL_DIR', ''),
                        SERVICES_CONFIG.get('RECOGNITION', {}).get('VOSK_MODELS', {}).get('en', '')
                    )
                    loading_tasks.append(
                        executor.submit(self.__load_speech_model, 'en', speech_model_path)
                    )

                    # Load translation models
                    for src_lang, tgt_lang in self.supported_translation_pairs:
                        loading_tasks.append(
                            executor.submit(self.__load_translation_model, src_lang, tgt_lang)
                        )
                        gc.collect()  # Clean memory after each model load

                    # Wait for all loading tasks to complete
                    for future in as_completed(loading_tasks):
                        try:
                            future.result()
                        except Exception as e:
                            self.logger.error(f"Model loading task failed: {str(e)}")
                            raise

            except Exception as e:
                self.logger.error(f"Model loading process failed: {str(e)}")
                raise

        def __load_speech_model(self, language: str, model_path: str):
            """Load speech recognition model into shared storage

            Args:
                language: Language code
                model_path: Path to model files
            """
            try:
                if language in self.__speech_models:
                    self.logger.info(f"Speech model already loaded for: {language}")
                    return

                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Speech model not found: {model_path}")

                model = Model(model_path)
                with self.__models_load_lock:
                    self.__speech_models[language] = model
                    self.logger.info(f"Speech model loaded: {language}")

            except Exception as e:
                self.logger.error(f"Speech model loading failed: {str(e)}")
                raise

        def __load_translation_model(self, src_lang: str, tgt_lang: str):
            """Load translation model into shared storage

            Args:
                src_lang: Source language code
                tgt_lang: Target language code
            """
            try:
                model_key = f"{src_lang}-{tgt_lang}"

                if model_key in self.__translation_pipelines:
                    self.logger.info(f"Translation pipeline already loaded for: {model_key}")
                    return

                model_name = ML_CONFIG.get('TRANSLATION', {}).get('MODEL_NAMES', {}).get(model_key)
                if not model_name:
                    raise ValueError(f"No model name found for language pair: {model_key}")

                # Set low_cpu_mem_usage to False to avoid requiring accelerate library
                use_low_memory = ML_CONFIG.get('TRANSLATION', {}).get('USE_LOW_MEMORY', True)

                # Check if accelerate is available
                try:
                    import accelerate
                    have_accelerate = True
                except ImportError:
                    have_accelerate = False
                    use_low_memory = False
                    self.logger.warning("Accelerate library not found, disabling low_cpu_mem_usage")

                translation_pipeline = pipeline(
                    "translation",
                    model=model_name,
                    model_kwargs={
                        "cache_dir": self.models_cache_dir,
                        "low_cpu_mem_usage": use_low_memory and have_accelerate
                    }
                )

                with self.__models_load_lock:
                    self.__translation_pipelines[model_key] = translation_pipeline
                    self.logger.info(f"Translation pipeline loaded: {model_key}")

            except Exception as e:
                self.logger.error(f"Translation pipeline loading failed: {str(e)}")
                raise

        def cleanup(self):
            """Clean up resources and models from shared storage"""
            with self.__models_load_lock:
                try:
                    # Clean speech models
                    for model in self.__speech_models.values():
                        if hasattr(model, '__del__'):
                            model.__del__()
                    self.__speech_models.clear()

                    # Clean translation pipelines
                    self.__translation_pipelines.clear()

                    # Reset loading state
                    self.__class__.__models_loaded = False
                    self.__loading_complete.clear()

                    # Force garbage collection
                    gc.collect()
                    self.logger.info("All models unloaded and memory cleaned")

                except Exception as e:
                    self.logger.error(f"Cleanup failed: {str(e)}")
                    raise

    def __init__(self):
        """Initialize LLMManager with required handlers"""
        self.logger = Logger()
        self.logger.info("Initializing LLM_Manager")

        # Initialize core handlers and services
        self.__model_loader = self.__ModelLoader()
        self.translation_handler = TranslationHandler()
        self.ocr_handler = OCRHandler()
        self.service = Services()

        # Supported prompt types
        self.prompts_supported = LLM_CONFIG.get('PROMPTS', {}).get('SUPPORTED', {})

        self.logger.info("LLM_Manager initialized successfully")

    def process_user_inputs(self, user_config, input_data):
        """Process and translate multimedia input based on user configuration"""
        self.logger.info(f"Starting translation with mode: {user_config['translation_mode']}")

        try:
            models_setup = self.__setup_translation_environment(user_config)
            if not models_setup:
                return None

            result = self.__process_input_by_mode(
                user_config['translation_mode'],
                input_data,
                models_setup
            )

            if result:
                self.logger.info("Translation completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return None

    def get_speech_model(self, src_language="en"):
        speech_model = self.__model_loader.get_speech_model(src_language)
        return speech_model

    def get_translation_model(self, src_language=None, tgt_language=None):
        translation_model = self.__model_loader.get_translation_model(src_language, tgt_language)
        return translation_model


    def __setup_translation_environment(self, user_config):
        """Setup translation environment and validate models"""
        src_code = self.service.get_language_code(user_config['source_lang'])
        dest_code = self.service.get_language_code(user_config['dest_lang'])

        if not src_code or not dest_code:
            self.logger.error("Invalid language configuration")
            return None

        translation_model = self.__model_loader.get_translation_model(src_code, dest_code)
        if not translation_model[2]:  # Check success flag
            self.logger.error("Failed to load translation model")
            return None

        speech_model = None
        if user_config['translation_mode'] in ['speech', 'both']:
            speech_model = self.__model_loader.get_speech_model(src_code)
            if not speech_model[1]:  # Check success flag
                self.logger.error("Failed to load speech model")
                return None
            speech_model = speech_model[0]  # Extract model from tuple

        return src_code, dest_code, translation_model[:2], speech_model

    def __process_input_by_mode(self, mode, input_data, models_setup):
        """Process input based on translation mode"""
        if not models_setup:
            return None

        src_code, _, translation_model, speech_model = models_setup
        try:
            if mode == 'image':
                return self.__process_image_input(input_data, src_code, translation_model)
            elif mode == 'speech':
                return self.__process_speech_input(input_data, speech_model, translation_model)
            elif mode == 'both':
                return self.__process_combined_input(input_data, src_code, speech_model, translation_model)
            else:
                self.logger.error(f"Invalid mode: {mode}")
                return None
        except ValueError as ve:
            self.logger.error(f"Input validation error: {str(ve)}")
            return None
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return None

    def __process_image_input(self, input_data, src_code, translation_model):
        """Handle image-only translation"""
        if not isinstance(input_data, np.ndarray):
            raise ValueError("Image mode requires array of pixels")

        result = self.__create_result_template()
        extracted_text = self.ocr_handler.extract_text_from_frame(input_data, src_code)

        if extracted_text:
            result['original_text'] = extracted_text
            result['translated_text'] = self.translation_handler.translate_text(
                extracted_text,
                translation_model
            )
            result['operation_type'] = 'translation'
            result['success'] = True
            result['extracted_text'] = extracted_text

        return result

    def __process_speech_input(self, input_data, speech_model, translation_model):
        """Handle speech-only translation"""
        if not isinstance(input_data, np.ndarray):
            raise ValueError("Speech mode requires numpy array of audio chunks")

        result = self.__create_result_template()
        recognized_text = self.service.recognize_text_from_speech(input_data, (speech_model, True))

        if recognized_text:
            result['original_text'] = recognized_text
            translated_text = self.translation_handler.translate_text(
                recognized_text,
                translation_model
            )
            if translated_text:
                result['translated_text'] = translated_text
                result['operation_type'] = 'translation'
                result['success'] = True

        return result

    def __process_combined_input(self, input_data, src_code, speech_model, translation_model):
        """Handle combined image and speech translation"""
        if not isinstance(input_data, tuple) or len(input_data) != 2:
            raise ValueError("Combined mode requires tuple of (frame_array, audio_chunks)")

        frame_array, audio_chunks = input_data
        if not isinstance(frame_array, np.ndarray) or not isinstance(audio_chunks, np.ndarray):
            raise ValueError("Invalid input data types")

        result = self.__create_result_template()

        # Process speech command
        recognized_text = self.service.recognize_text_from_speech(audio_chunks, (speech_model, True))
        if not recognized_text:
            return result

        result['original_text'] = recognized_text
        command = self.service.verify_user_input(recognized_text, self.prompts_supported)

        # Process image
        extracted_text = self.ocr_handler.extract_text_from_frame(frame_array, src_code)
        if not extracted_text:
            return result

        result['extracted_text'] = extracted_text

        if command == "tr":
            translated_text = self.translation_handler.translate_text(
                extracted_text,
                translation_model
            )
            if translated_text:
                result['translated_text'] = translated_text
                result['operation_type'] = 'translation'
                result['success'] = True
        elif command == "ex":
            result['operation_type'] = 'extraction'
            result['success'] = True
        else:
            self.logger.warning(f"Unsupported command: {command}")
            result['success'] = False
            result['operation_type'] = None

        return result

    def __create_result_template(self):
        """Create empty result template"""
        return {
            'success': False,
            'operation_type': None,
            'original_text': '',
            'translated_text': '',
            'extracted_text': None
        }

    def cleanup(self):
        """Clean up resources and models"""
        try:
            self.__model_loader.cleanup()
            self.logger.info("Successfully cleaned up LLM Manager resources")
        except Exception as e:
            self.logger.error(f"Failed to clean up resources: {str(e)}")
            raise
