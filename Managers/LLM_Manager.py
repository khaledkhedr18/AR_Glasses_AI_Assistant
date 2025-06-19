from utils.Logging import Logger
from Handlers.OCR_Handler import OCRHandler
from Handlers.Translation_Handler import TranslationHandler
from Handlers.LTD_Handler import LTDHandler
from utils.Config import LLM_CONFIG
from utils.Services import Services, IO_CONFIG
import numpy as np
from datetime import datetime
import threading


class LLMManager:
    """
    Central manager for all language model operations.

    This class acts as a communication hub for:
    1. LLM handlers to communicate with each other indirectly
    2. External managers (like IO_Manager) to access language model functionality
    """

    def __init__(self):
        """
        Initialize the LLM_Manager with all required handlers.
        """
        self.logger = Logger()
        self.logger.info("Initializing LLM_Manager")

        self.supported_lang_codes = LLM_CONFIG['SUPPORTED_LANG_CODES']
        self.supported_languages = LLM_CONFIG['SUPPORTED_LANGUAGES']
        self.supported_modes = IO_CONFIG['SUPPORTED_MODES']
        self.prompts_supported = LLM_CONFIG['PROMPTS_SUPPORTED']
        # Initialize core handlers
        self.translation_handler = TranslationHandler()
        self.ocr_handler = OCRHandler()
        self.ltd_handler = LTDHandler()
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

        self.logger.info("LLM_Manager initialized successfully")

    def process_multimedia_input(self, user_config, input_data):
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

        src_lang_code = self._get_language_code(user_config['source_lang'])
        dest_lang_code = self._get_language_code(user_config['dest_lang'])

        if not src_lang_code or not dest_lang_code:
            self.logger.error("Invalid language configuration")
            return None

        try:
            self.ltd_handler.load_language_model(src_lang_code)
            self.ltd_handler.load_language_model(dest_lang_code)
        except Exception as e:
            self.logger.error(f"Failed to load language models: {str(e)}")
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
                    result['translated_text'] = self.translation_handler.translate_text(
                        extracted_text, src_lang_code, dest_lang_code
                    )
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
                    speech_text = self.translation_handler.translate_text(
                        recognized_text, src_lang_code, dest_lang_code
                    )
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
                            image_translation = self.translation_handler.translate_text(
                                extracted_text, src_lang_code, dest_lang_code
                            )
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

    def _get_language_code(self, language):
        """
        Convert language name to language code.

        Args:
            language (str): Language name or code

        Returns:
            str: Language code or None if invalid
        """
        # If it's already a code, return it
        if language in  self.supported_lang_codes:
            return language

        return self.supported_languages.get(language.lower(), None)

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

    def process_text_command(self, text_input):
        """
        Process a text command from the user.


        Args:
            text_input (str): Text command from user

        Returns:
            dict: Response with type and content
        """
        self.logger.info(f"Processing text command: '{text_input}'")

        # Determine command type
        command_type = self._determine_command_type(text_input)

        if command_type == "translation":
            return self._handle_translation_command(text_input)
        elif command_type == "ocr":
            return self._handle_ocr_command()
        else:
            # Default to general response
            return {
                "type": "text",
                "content": f"I understand you said: {text_input}"
            }


    def extract_text_from_image(self, image_path, language=None):
        """
        Extract text from an image.

        Args:
            image_path (str): Path to image file
            language (str, optional): Language of text in the image

        Returns:
            str: Extracted text or error message
        """
        self.logger.info(f"Extracting text from image: {image_path}")

        # Use default language if not specified
        language = language or self.default_source_lang
        language_code = self._get_language_code(language)

        if not language_code:
            self.logger.error(f"Invalid language specified: {language}")
            return None

        # Use OCR handler
        text = self.ocr_handler.get_text_from_image(image_path, language_code)

        if text:
            self.logger.info(f"OCR result: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            return text
        else:
            self.logger.error("OCR failed or no text found")
            return None












    def process_image_and_translate(self, image_path, source_lang=None, target_lang=None):
        """
        Process an image, extract text, and translate it.

        Args:
            image_path (str): Path to image file
            source_lang (str, optional): Source language
            target_lang (str, optional): Target language

        Returns:
            dict: Results with original and translated text
        """
        self.logger.info(f"Processing image and translating: {image_path}")

        # Extract text from image
        source_lang = source_lang or self.default_source_lang
        source_code = self._get_language_code(source_lang)
        extracted_text = self.extract_text_from_image(image_path, source_code)

        if not extracted_text:
            return {
                "success": False,
                "error": "No text could be extracted from the image"
            }

        # Translate the extracted text
        target_lang = target_lang or self.default_target_lang
        target_code = self._get_language_code(target_lang)

        translated_text = self.translate_text(extracted_text, source_code, target_code)

        if not translated_text:
            return {
                "success": False,
                "original_text": extracted_text,
                "error": "Translation failed"
            }

        return {
            "success": True,
            "original_text": extracted_text,
            "translated_text": translated_text,
            "source_language": source_lang,
            "target_language": target_lang
        }

    # ======== PRIVATE HELPER METHODS ========

    def _determine_command_type(self, text):
        """
        Determine the type of command from text input.

        Args:
            text (str): User input text

        Returns:
            str: Command type ("translation", "ocr", "general")
        """
        text_lower = text.lower()

        # Check for translation indicators
        if any(word in text_lower for word in ["translate", "translation", "convert language"]):
            return "translation"

        # Check for OCR indicators
        elif any(phrase in text_lower for phrase in ["read text", "extract text", "ocr", "read image"]):
            return "ocr"

        # Default to general command
        return "general"

    def _handle_translation_command(self, text_input):
        """
        Handle a translation command.

        Args:
            text_input (str): Text command

        Returns:
            dict: Response data
        """
        # The workflow here would depend on your specific requirements
        # This is a simplified example
        return {
            "type": "translation_request",
            "content": "Please specify source and target languages"
        }

    def _handle_ocr_command(self):
        """
        Handle an OCR command.

        Returns:
            dict: Response data
        """
        return {
            "type": "ocr_request",
            "content": "Ready to capture image for text extraction"
        }



    def cleanup(self):
        """Clean up resources used by the LLM_Manager and its handlers."""
        self.logger.info("Cleaning up LLM_Manager resources")

        # Clean up translation resources
        if hasattr(self.translation_handler, 'release_translation_resources'):
            self.translation_handler.release_translation_resources()

        # Clear speech models
        with self._model_lock:
            self.speech_models.clear()



