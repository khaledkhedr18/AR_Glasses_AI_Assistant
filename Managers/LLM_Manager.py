import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
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
        Integrated simple model loading approach"""

        # Class-level variables for shared resources
        __models_load_lock = threading.Lock()
        __vosk_models = {}  # Stores speech recognition models
        __translation_pipelines = {}  # Stores translation pipelines

        def __init__(self):
            """Initialize handler instance"""
            self.service = Services()
            self.logger = Logger()
            self.logger.info("Initializing Model Loader Handler")

            # Core configuration initialization
            self.__load_config()

        def __load_config(self):
            """Load configuration settings for models"""
            translation_config = ML_CONFIG.get('TRANSLATION', {})
            recognition_config = SERVICES_CONFIG.get('RECOGNITION', {})

            # Initialize configuration attributes with default values and expand paths
            self.models_cache_dir = os.path.expanduser(translation_config.get('MODELS_CACHE_DIR', './models/cache'))
            self.models_dir = os.path.expanduser(translation_config.get('TRANSLATION_MODELS_DIR', './models/translation'))
            self.models_load_timeout = translation_config.get('MODELS_LOAD_TIMEOUT', 60)
            self.supported_translation_pairs = translation_config.get('SUPPORTED_TRANSLATION_PAIRS', [])
            self.supported_speech_models = translation_config.get('SUPPORTED_SPEECH_MODELS', [])

            # Expand the tilde in the path
            self.recognizer_model_path = os.path.expanduser(recognition_config.get('VOSK_MODEL_DIR', './models/vosk'))

        def get_vosk_model(self, language_code: str):
            """
            Get the Vosk model for the given language
            Loads it if not already loaded
            """
            with self.__models_load_lock:
                if language_code not in self.__vosk_models:
                    try:
                        if language_code == "ar":
                            model_path = "models/vosk/vosk-model-ar-mgb2-0.4"
                        elif language_code == "en":
                            model_path = "models/vosk/vosk-model-small-en-us-0.15"
                        elif language_code == "fr":
                            model_path = "models/vosk/vosk-model-small-fr-0.22"
                        else:
                            raise ValueError(f"Unsupported Language Code: {language_code}")

                        # Use absolute path
                        model_path = os.path.abspath(model_path)
                        if not os.path.exists(model_path):
                            raise FileNotFoundError(f"Model not found at: {model_path}")

                        self.logger.info(f"Loading VOSK model for {language_code} from {model_path}")

                        self.__vosk_models[language_code] = Model(model_path)

                        self.logger.info(f"VOSK model for {language_code} loaded successfully")

                    except Exception as e:
                        self.logger.error(f"Failed to load VOSK model for {language_code}: {str(e)}")
                        raise

                return self.__vosk_models[language_code]

        def get_translation_pipeline(self, source: str, target: str):
            """Get translation pipeline with optimized settings"""
            key = f"{source}-{target}"
            if key not in self.__translation_pipelines:
                try:
                    model_name = f'Helsinki-NLP/opus-mt-{source}-{target}'

                    # Add these critical pipeline parameters
                    self.__translation_pipelines[key] = pipeline(
                        "translation",
                        model=model_name,
                        device=-1,  # CPU
                        truncation=True,  # Essential for short texts
                        max_length=100,  # Prevent long outputs
                        num_beams=2,  # Balance between speed and quality
                        early_stopping=True  # Prevent repetition
                    )
                except Exception as e:
                    self.logger.error(f"Pipeline creation failed: {str(e)}")
                    raise
            return self.__translation_pipelines[key]
        def get_translation_model(self, source_lang: str, target_lang: str) -> tuple:
            """Get translation pipeline using integrated model loader

            Args:
                source_lang: Source language code
                target_lang: Target language code

            Returns:
                tuple: (pipeline, tokenizer, success_flag)
            """
            try:
                self.logger.info(f"Loading translation model on demand: {source_lang}-{target_lang}")
                pipeline_obj = self.get_translation_pipeline(source_lang, target_lang)
                return (pipeline_obj, None, True)
            except Exception as e:
                self.logger.error(f"Failed to load translation model {source_lang}-{target_lang}: {str(e)}")
                return (None, None, False)

        def get_speech_model(self, language: str) -> tuple:
            """Get speech model using integrated model loader

            Args:
                language: Language code

            Returns:
                tuple: (model, success_flag)
            """
            try:
                self.logger.info(f"Loading speech model on demand: {language}")
                model = self.get_vosk_model(language)
                return (model, True)
            except Exception as e:
                self.logger.error(f"Failed to load speech model {language}: {str(e)}")
                return (None, False)

        def cleanup(self):
            """Clean up resources and models from shared storage"""
            with self.__models_load_lock:
                try:
                    # Clean speech models
                    for model in self.__vosk_models.values():
                        if hasattr(model, '__del__'):
                            model.__del__()
                    self.__vosk_models.clear()

                    # Clean translation pipelines
                    self.__translation_pipelines.clear()

                    self.logger.info("All models cleaned up successfully")
                except Exception as e:
                    self.logger.error(f"Error during cleanup: {str(e)}")
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
        """Process and translate multimedia input with complete error handling"""
        self.logger.info(f"Starting translation with mode: {user_config.get('translation_mode')}")

        try:
            # Setup translation environment
            models_setup = self.__setup_translation_environment(user_config)
            if not models_setup:
                return {
                    'success': False,
                    'error': 'Failed to setup translation environment',
                    'operation_type': None,
                    'original_text': '',
                    'translated_text': ''
                }

            # Process based on mode
            result = self.__process_input_by_mode(
                user_config['translation_mode'],
                input_data,
                models_setup
            )

            if not result:
                result = self.__create_result_template()
                result['error'] = 'Translation process failed'
            elif not result.get('success'):
                result['error'] = result.get('error', 'Unknown translation error')

            return result

        except Exception as e:
            self.logger.error(f"Translation processing error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'operation_type': None,
                'original_text': '',
                'translated_text': ''
            }
    def get_speech_model(self, src_language="en"):
        speech_model = self.__model_loader.get_speech_model(src_language)
        return speech_model

    def get_translation_model(self, src_language=None, tgt_language=None):
        translation_model = self.__model_loader.get_translation_model(src_language, tgt_language)
        return translation_model


    def __setup_translation_environment(self, user_config):
        """Setup translation environment with robust configuration validation"""
        required_keys = ['source_lang', 'target_lang', 'translation_mode']
        if not all(key in user_config for key in required_keys):
            missing = [key for key in required_keys if key not in user_config]
            self.logger.error(f"Missing configuration keys: {missing}")
            return None

        # Validate source language
        src_code = self.service.get_language_code(user_config['source_lang'])
        if not src_code:
            self.logger.error(f"Invalid source language: {user_config['source_lang']}")
            return None

        # Validate target language
        dest_code = self.service.get_language_code(user_config['target_lang'])
        if not dest_code:
            self.logger.error(f"Invalid target language: {user_config['target_lang']}")
            return None

        # Check supported translation pairs
        supported_pairs = ML_CONFIG.get('TRANSLATION', {}).get('SUPPORTED_TRANSLATION_PAIRS', [])
        if (src_code, dest_code) not in supported_pairs:
            self.logger.error(f"Unsupported translation pair: {src_code}-{dest_code}")
            return None

        # Load translation model
        translation_result = self.__model_loader.get_translation_model(src_code, dest_code)
        if not translation_result or len(translation_result) < 3 or not translation_result[2]:
            self.logger.error(f"Failed to load translation model for {src_code}-{dest_code}")
            return None

        # Load speech model if needed
        speech_model = None
        if user_config['translation_mode'] in ['speech', 'both']:
            speech_result = self.__model_loader.get_speech_model(src_code)
            if not speech_result or len(speech_result) < 2 or not speech_result[1]:
                self.logger.error(f"Failed to load speech model for {src_code}")
                return None
            speech_model = speech_result[0]

        return {
            'source_code': src_code,
            'target_code': dest_code,
            'translation_pipeline': translation_result[0],  # The actual pipeline
            'translation_tokenizer': translation_result[1],  # Tokenizer if available
            'speech_model': speech_model,
            'translation_mode': user_config['translation_mode']
        }

    def __process_input_by_mode(self, mode, input_data, models_setup):
        if not models_setup:
            return None

        try:
            if mode == 'image':
                return self.__process_image_input(
                    input_data,
                    models_setup['source_code'],
                    models_setup['translation_pipeline']
                )
            elif mode == 'text':  # New text mode handler
                return self.__process_text_input(
                    input_data,
                    models_setup['translation_pipeline']
                )
            elif mode == 'speech':
                return self.__process_speech_input(
                    input_data,
                    models_setup['speech_model'],
                    models_setup['translation_pipeline']
                )
            elif mode == 'both':
                return self.__process_combined_input(
                    input_data,
                    models_setup['source_code'],
                    models_setup['speech_model'],
                    models_setup['translation_pipeline']
                )
            else:
                self.logger.error(f"Invalid mode: {mode}")
                return None
        except Exception as e:
            self.logger.error(f"Processing error: {str(e)}")
            return None

    def __process_text_input(self, text, translation_model):
        """Handle direct text translation with proper logging"""
        result = self.__create_result_template()
        self.logger.info(f"Starting text translation for: {text}")

        if not text:
            self.logger.error("No text provided for translation")
            return result

        try:
            self.logger.info("Calling translation pipeline...")
            translated_text = self.translation_handler.translate_text(
                text,
                translation_model
            )

            if translated_text:
                self.logger.info(f"Translation successful: {translated_text}")
                result.update({
                    'success': True,
                    'operation_type': 'translation',
                    'original_text': text,
                    'translated_text': translated_text
                })
            else:
                self.logger.error("Translation returned empty result")

        except Exception as e:
            self.logger.error(f"Translation process failed: {str(e)}")

        return result

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
