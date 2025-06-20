from utils.Logging import Logger
from Handlers.OCR_Handler import OCRHandler
from Handlers.Translation_Handler import TranslationHandler
from Handlers.LTD_Handler import LTDHandler
from utils.Config import LLM_CONFIG, SERVICES_CONFIG
from utils.Services import Services
import numpy as np
from datetime import datetime
import threading


class LLMManager:
    """
    Central manager for all language model operations.

    This class acts as a communication hub for:
    1. LLM handlers to communicate with each other indirectly
    2. External managers to access language model functionality
    """

    def __init__(self):
        """
        Initialize the LLM_Manager with all required handlers.
        """
        self.logger = Logger()
        self.logger.info("Initializing LLM_Manager")

        # Initialize core handlers
        self.ltd_handler = LTDHandler()
        self.translation_handler = TranslationHandler()
        self.ocr_handler = OCRHandler()
        self.service = Services()

        # Mutex lock for thread-safe model loading
        self._model_lock = threading.Lock()

        # Model and resource caches
        self.active_translations = {}
        self.translation_counter = 0

        # Performance settings
        self.ocr_batch_size = 1024  # bytes
        self.translation_batch_size = 2048  # characters

        self.supported_lang_codes = SERVICES_CONFIG['SUPPORTED_LANG_CODES']
        self.supported_languages = SERVICES_CONFIG['LANGUAGES_MAP']
        self.prompts_supported = LLM_CONFIG['PROMPTS_SUPPORTED']

        self.logger.info("LLM_Manager initialized successfully")

    def process_user_inputs(self, user_config, input_data):
        """
        Process and translate multimedia input based on user configuration.

        Args:
            user_config (dict): {
                'source_lang': source language,
                'dest_lang': destination language,
                'translation_mode': 'image'|'speech'|'both'
            }
            input_data: Can be:
                - numpy.ndarray: Array of pixels from capture_array()
                - numpy.ndarray: Audio wave chunks
                - tuple: (frame_array, audio_chunks) for combined mode

        Returns:
            dict: Translation results with extracted and translated content
        """
        self.logger.info(f"Starting translation with mode: {user_config['translation_mode']}")

        try:
            # Validate and setup models
            models_setup = self.__setup_translation_environment(user_config)
            if not models_setup:
                return None

            # Process the input based on mode
            result = self.__process_input_by_mode(user_config['translation_mode'], input_data, models_setup)
            if not result:
                return None

            self.logger.info("Translation completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return None

    def get_all_translations(self):
        """
        Retrieve all stored translations.

        Returns:
            dict: All stored translations
        """
        return self.active_translations

    def clear_stored_translations(self):
        """
        Clear all stored translations from memory.
        """
        self.active_translations.clear()
        self.translation_counter = 0

    def get_stored_translation(self, index):
        """
        Retrieve a specific stored translation by index.

        Args:
            index (int): The translation index

        Returns:
            dict: Translation data or None if not found
        """
        return self.active_translations.get(index)

    def __setup_translation_environment(self, user_config):
        """
        Setup and validate the translation environment including models.

        Args:
            user_config (dict): User configuration with language settings

        Returns:
            tuple: (src_code, dest_code, translation_model, speech_model) or None if setup fails
        """
        # Get language codes
        src_code = self.service.get_language_code(user_config['source_lang'])
        dest_code = self.service.get_language_code(user_config['dest_lang'])

        if not src_code or not dest_code:
            self.logger.error("Invalid language configuration")
            return None

        # Get translation model
        translation_model = self.ltd_handler.get_translation_model(src_code, dest_code)
        if not translation_model or not translation_model[0]:
            self.logger.error("Failed to load translation model")
            return None

        # Get speech model if needed
        speech_model = None
        if user_config['translation_mode'] in ['speech', 'both']:
            speech_model = self.ltd_handler.get_speech_model(src_code)
            if not speech_model or not speech_model[0]:
                self.logger.error("Failed to load speech model")
                return None

        return src_code, dest_code, translation_model, speech_model

    def __process_input_by_mode(self, mode, input_data, models_setup):
        """
        Process input based on translation mode.

        Args:
            mode (str): Translation mode
            input_data: Input data to process
            models_setup (tuple): (src_code, dest_code, translation_model, speech_model)

        Returns:
            dict: Translation results or None if processing fails
        """
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
        """Handle image-only translation."""
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
        """Handle speech-only translation."""
        if not isinstance(input_data, np.ndarray):
            raise ValueError("Speech mode requires numpy array of audio chunks")

        result = self.__create_result_template()
        recognized_text = self.service.recognize_text_from_speech(input_data, speech_model)

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
        """Handle combined image and speech translation."""
        if not isinstance(input_data, tuple) or len(input_data) != 2:
            raise ValueError("Combined mode requires tuple of (frame_array, audio_chunks)")

        frame_array, audio_chunks = input_data
        if not isinstance(frame_array, np.ndarray) or not isinstance(audio_chunks, np.ndarray):
            raise ValueError("Invalid input data types")

        result = self.__create_result_template()

        # Process speech command
        recognized_text = self.service.recognize_text_from_speech(audio_chunks, speech_model)
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
            return self.__handle_translation_command(extracted_text, translation_model)
        elif command == "ex":
            return self.__handle_extraction_command(extracted_text, src_code)

        return result

    def __handle_translation_command(self, extracted_text, translation_model):
        """Process translation command in combined mode."""
        result = self.__create_result_template()
        translated_text = self.translation_handler.translate_text(extracted_text, translation_model)

        if translated_text:
            result['translated_text'] = translated_text
            result['operation_type'] = 'translation'
            result['success'] = True
            result['extracted_text'] = extracted_text

        return result

    def __handle_extraction_command(self, extracted_text, src_code):
        """Process extraction command in combined mode."""
        result = self.__create_result_template()

        self.active_translations[self.translation_counter] = {
            'text': extracted_text,
            'timestamp': datetime.now(),
            'source_lang': src_code
        }

        result['storage_index'] = self.translation_counter
        result['operation_type'] = 'extraction'
        result['success'] = True
        result['extracted_text'] = extracted_text

        self.translation_counter += 1
        return result

    def __create_result_template(self):
        """Create empty result template."""
        return {
            'success': False,
            'operation_type': None,
            'original_text': '',
            'translated_text': '',
            'extracted_text': None,
            'storage_index': None
        }