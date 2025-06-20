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
        self.language_models = {}
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
        Process and translate multimedia input (images and/or speech) based on user configuration.

        Args:
            user_config (dict): {
                'source_lang': source language,
                'dest_lang': destination language,
                'translation_mode': 'image'|'speech'|'both'
            }
            input_data: Can be:
                - numpy.ndarray: Array of pixels from picam2.capture_array()
                - numpy.ndarray: Audio wave chunks
                - tuple: (frame_array, audio_chunks) for combined mode

        Returns:
            dict: Translation results with extracted and translated content
        """
        self.logger.info(f"Starting translation with mode: {user_config['translation_mode']}")

        src_lang_code = self.service.get_language_code(user_config['source_lang'])
        dest_lang_code = self.service.get_language_code(user_config['dest_lang'])

        if not src_lang_code or not dest_lang_code:
            self.logger.error("Invalid language configuration")
            return None

        model_components = self.ltd_handler.get_translation_model(src_lang_code, dest_lang_code)
        if not model_components or not model_components[0]:
            self.logger.error("Failed to retrieve translation model components")
            return None

        result = {
            'success': False,
            'operation_type': None,  # Will be 'translation' or 'extraction'
            'original_text': '',
            'translated_text': '',
            'extracted_text': None,
            'storage_index': None  # Will store the index if text was saved
        }

        try:
            if user_config['translation_mode'] == 'image':
                if not isinstance(input_data, np.ndarray):
                    raise ValueError("Image mode requires array of pixels from capture_array()")

                # First extract text from frame
                extracted_text = self.ocr_handler.extract_text_from_frame(input_data, src_lang_code)
                result['extracted_text'] = extracted_text

                if extracted_text:
                    result['original_text'] = extracted_text
                    # Then translate if text was found
                    result['translated_text'] = self.translation_handler.translate_text(extracted_text, model_components)
                    result['operation_type'] = 'translation'
                    result['success'] = True

            elif user_config['translation_mode'] == 'speech':
                if not isinstance(input_data, np.ndarray):
                    raise ValueError("Speech mode requires numpy array of audio chunks")

                # First recognize speech
                recognized_text = self.service.recognize_text_from_speech(input_data, src_lang_code)
                if recognized_text:
                    result['original_text'] = recognized_text
                    # Then translate recognized text
                    speech_text = self.translation_handler.translate_text(recognized_text, model_components)
                    if speech_text:
                        result['translated_text'] = speech_text
                        result['operation_type'] = 'translation'
                        result['success'] = True

            elif user_config['translation_mode'] == 'both':
                if not isinstance(input_data, tuple) or len(input_data) != 2:
                    raise ValueError("Both mode requires tuple of (frame_array, audio_chunks)")

                frame_array, audio_chunks = input_data
                if not isinstance(frame_array, np.ndarray) or not isinstance(audio_chunks, np.ndarray):
                    raise ValueError("Both inputs must be numpy arrays")

                # First handle speech recognition
                recognized_text = self.service.recognize_text_from_speech(audio_chunks, src_lang_code)
                if recognized_text:
                    result['original_text'] = recognized_text
                    # Get command from speech ("tr" or "ex")
                    command = self.service.verify_user_input(recognized_text, self.prompts_supported)

                    # Extract text from image
                    extracted_text = self.ocr_handler.extract_text_from_frame(frame_array, src_lang_code)

                    # Then handle image content translation
                    if extracted_text:
                        result['extracted_text'] = extracted_text

                        # If command is "tr", translate the extracted text
                        if command == "tr":
                            image_translation = self.translation_handler.translate_text(extracted_text, model_components)
                            if image_translation:
                                result['translated_text'] = image_translation
                                result['operation_type'] = 'translation'
                                result['success'] = True

                        # If command is "ex", just store the extracted text
                        elif command == "ex":
                            # Store in memory/cache with metadata
                            self.active_translations[self.translation_counter] = {
                                'text': extracted_text,
                                'timestamp': datetime.now(),
                                'source_lang': src_lang_code
                            }
                            result['storage_index'] = self.translation_counter
                            result['operation_type'] = 'extraction'
                            result['success'] = True
                            result['extracted_text'] = extracted_text
                            self.translation_counter += 1

        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return None

        self.logger.info("Translation completed" if result['success'] else "Translation failed")
        return result if result['success'] else None

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
