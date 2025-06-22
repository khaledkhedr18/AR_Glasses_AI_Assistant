import os
import gc
import torch
import threading
import numpy as np
from vosk import Model
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import MarianMTModel, MarianTokenizer
from Handlers.OCR_Handler import OCRHandler
from Handlers.Translation_Handler import TranslationHandler
from utils.config import ML_CONFIG, LLM_CONFIG, SERVICES_CONFIG
from utils.Services import Services
from utils.Logging import Logger


class LLMManager:
    """
    Central manager for all language model operations.
    Handles translation and extraction of text from images and speech.
    """

    class __ModelLoader:
        """Private model loader handler for managing ML models.
        Implements singleton pattern to prevent multiple model loading."""

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
                        cls._instance = super(LLMManager.__ModelLoader, cls).__new__(cls)
            return cls._instance

        def __init__(self):
            """Initialize handler with memory-optimized settings (thread-safe)"""
            if self._models_initialized:
                return

            with self._initialization_lock:
                if not self._models_initialized:
                    self.logger = Logger()
                    self.logger.info("Initializing Model Loader Handler")

                    # Core components initialization
                    self.service = Services()
                    # Updated to dictionary access
                    self.models_dir = ML_CONFIG['TRANSLATION']['TRANSLATION_MODELS_DIR']
                    self.recognizer_model_path = SERVICES_CONFIG['RECOGNITION']['RECOGNITION_MODEL_DIR']
                    self.model_timeout = ML_CONFIG['TRANSLATION']['MODELS_LOAD_TIMEOUT']

                    # Load models only if not already loaded
                    self.__load_models()
                    self._models_initialized = True
                    self.logger.info("Model Loader Handler initialization completed")

        def get_translation_model(self, source_lang, target_lang):
            """Get translation model from cache"""
            if not self._loading_complete.wait(timeout=self.model_timeout):
                self.logger.error("Model loading timeout exceeded")
                return None, None, False

            src_code = self.service.get_language_code(source_lang)
            tgt_code = self.service.get_language_code(target_lang)

            if not src_code or not tgt_code:
                self.logger.error(f"Invalid language codes: {source_lang}, {target_lang}")
                return None, None, False

            pair = (src_code, tgt_code)
            # Updated to dictionary access
            if pair not in ML_CONFIG['TRANSLATION']['SUPPORTED_TRANSLATION_PAIRS']:
                self.logger.error(f"Unsupported translation pair: {pair}")
                return None, None, False

            model_key = f"{src_code}-{tgt_code}"
            model_pair = self._translation_models.get(model_key)

            return (*model_pair, True) if model_pair else (None, None, False)

        def get_speech_model(self, language):
            """Get speech model from cache"""
            if not self._loading_complete.wait(timeout=self.model_timeout):
                self.logger.error("Model loading timeout exceeded")
                return None, False

            lang_code = self.service.get_language_code(language)
            # Updated to dictionary access
            if not lang_code or lang_code not in ML_CONFIG['TRANSLATION']['SUPPORTED_SPEECH_MODELS']:
                self.logger.error(f"Unsupported speech language: {language}")
                return None, False

            model = self._speech_models.get(lang_code)
            return (model, True) if model else (None, False)

        def __load_models(self):
            """Initialize and load models if not already loaded (thread-safe)"""
            if self._loading_complete.is_set():
                self.logger.info("Models already loaded, using existing models")
                return

            with self._initialization_lock:
                if not self._loading_complete.is_set():
                    try:
                        gc.collect()
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None

                        self.__ensure_model_directories()
                        self.__preload_all_models()

                        self._loading_complete.set()
                        self.logger.info("Initial model loading completed successfully")
                    except Exception as e:
                        self.logger.error(f"Model loading failed: {str(e)}")
                        self._loading_complete.clear()
                        raise

        def __ensure_model_directories(self):
            """Create and verify required model directories"""
            directories = [
                self.models_dir,
                self.recognizer_model_path,
                # Updated to dictionary access
                ML_CONFIG['TRANSLATION']['MODELS_DIR']
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
                # Updated to dictionary access
                with ThreadPoolExecutor(max_workers=ML_CONFIG['TRANSLATION']['MAX_WORKERS']) as executor:
                    speech_model_path = os.path.join(
                        # Updated to dictionary access
                        SERVICES_CONFIG['RECOGNITION']['VOSK_MODEL_PATH'],
                        SERVICES_CONFIG['RECOGNITION']['VOSK_MODELS']['en']
                    )
                    loading_tasks.append(
                        executor.submit(self.__load_speech_model, 'en', speech_model_path)
                    )

                    # Updated to dictionary access
                    for src_lang, tgt_lang in ML_CONFIG['TRANSLATION']['SUPPORTED_TRANSLATION_PAIRS']:
                        loading_tasks.append(
                            executor.submit(
                                self.__load_translation_model,
                                src_lang,
                                tgt_lang
                            )
                        )
                        gc.collect()

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
            """Load speech recognition model"""
            try:
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
            """Load translation model"""
            try:
                model_key = f"{src_lang}-{tgt_lang}"

                if model_key in self._translation_models:
                    self.logger.info(f"Translation model already loaded for: {model_key}")
                    return

                # Updated to dictionary access
                model_name = ML_CONFIG['TRANSLATION']['MODEL_NAMES'].get(model_key)
                if not model_name:
                    raise ValueError(f"No model name found for language pair: {model_key}")

                # Updated to dictionary access for torch_dtype and cache_dir
                torch_dtype_str = ML_CONFIG['TRANSLATION']['TORCH_DTYPE']
                model = MarianMTModel.from_pretrained(
                    model_name,
                    torch_dtype=getattr(torch, torch_dtype_str),
                    cache_dir=ML_CONFIG['TRANSLATION']['MODELS_DIR'],
                    low_cpu_mem_usage=True,
                    return_dict=False
                )

                tokenizer = MarianTokenizer.from_pretrained(
                    model_name,
                    cache_dir=ML_CONFIG['TRANSLATION']['MODELS_DIR'],
                    model_max_length=512
                )

                with self._initialization_lock:
                    self._translation_models[model_key] = (model, tokenizer)
                    self.logger.info(f"Translation model loaded: {model_key}")

            except Exception as e:
                self.logger.error(f"Translation model loading failed: {str(e)}")
                raise

        def cleanup(self):
            """Clean up resources"""
            with self._initialization_lock:
                try:
                    for model in list(self._speech_models.values()):
                        if hasattr(model, '__del__'):
                            model.__del__()
                    self._speech_models.clear()

                    for model_pair in list(self._translation_models.values()):
                        if model_pair:
                            del model_pair[0]
                            del model_pair[1]
                    self._translation_models.clear()

                    self.__class__._models_initialized = False
                    self._loading_complete.clear()

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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

        # Supported prompt types - Updated to dictionary access
        self.prompts_supported = LLM_CONFIG['PROMPTS']['SUPPORTED']

        self.logger.info("LLM_Manager initialized successfully")

    # Rest of the LLMManager methods remain unchanged
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
