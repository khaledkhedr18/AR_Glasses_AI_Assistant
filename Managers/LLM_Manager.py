from utils.Logging import Logger
from Handlers.OCR_Handler import OCRHandler
from Handlers.Translation_Handler import TranslationHandler
from Handlers.LTD_Handler import LTDHandler
from utils.Config import LLM_CONFIG, IO_CONFIG
from utils.Services import Services
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

        # Translation configurations
        self.supported_modes = ['image', 'speech', 'both']
        self.supported_languages = {
            'english': 'en',
            'arabic': 'ar',
            'french': 'fr'
            # Add more languages as needed
        }

        # Performance settings
        self.ocr_batch_size = 1024  # bytes
        self.translation_batch_size = 2048  # characters

        self.logger.info("LLM_Manager initialized successfully")

    def translate_input(self, user_config, input_data):
        """
        Translate based on input type and user configuration.

        Args:
            user_config (dict): {
                'source_lang': source language,
                'dest_lang': destination language,
                'translation_mode': 'image'|'speech'|'both'
            }
            input_data: Can be:
                - str: Path to image file
                - bytes: Audio data
                - tuple: (image_path, audio_data) for combined mode

        Returns:
            dict: Translation results
        """
        self.logger.info(f"Starting translation with mode: {user_config['translation_mode']}")

        source_code = self._get_language_code(user_config['source_lang'])
        dest_code = self._get_language_code(user_config['dest_lang'])

        if not source_code or not dest_code:
            self.logger.error("Invalid language configuration")
            return None

        try:
            self.ltd_handler.load_language_model(source_code)
            self.ltd_handler.load_language_model(dest_code)
        except Exception as e:
            self.logger.error(f"Failed to load language models: {str(e)}")
            return None

        result = {
            'success': False,
            'original_text': '',
            'translated_text': '',
            'speech_translation': None
        }

        try:
            if user_config['translation_mode'] == 'image':
                if not isinstance(input_data, str):
                    raise ValueError("Image mode requires image path string")

                extracted_text = self.ocr_handler.recognize_text_from_image(input_data, source_code)
                if extracted_text:
                    result['original_text'] = extracted_text
                    result['translated_text'] = self.translation_handler.translate_text(extracted_text, source_code, dest_code)
                    result['success'] = bool(result['translated_text'])

            elif user_config['translation_mode'] == 'speech':
                if not isinstance(input_data, bytes):
                    raise ValueError("Speech mode requires audio data bytes")

                speech_text = self.translation_handler.translate_text(input_data, source_code, dest_code)
                if speech_text:
                    result['original_text'] = input_data
                    result['translated_text'] = speech_text
                    result['success'] = True

            elif user_config['translation_mode'] == 'both':
                if not isinstance(input_data, tuple) or len(input_data) != 2:
                    raise ValueError("Both mode requires tuple of (image_path, audio_data)")

                image_path, audio_data = input_data

                # Process image
                extracted_text = self.ocr_handler.recognize_text_from_image(image_path, source_code)
                if extracted_text:
                    result['original_text'] = extracted_text
                    result['translated_text'] = self.translation_handler.translate_text(extracted_text, source_code, dest_code)
                    result['success'] = bool(result['translated_text'])

                # Process speech
                speech_translation = self.translation_handler.translate_text(audio_data, source_code, dest_code)
                if speech_translation:
                    result['speech_translation'] = speech_translation

        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return None

        self.logger.info("Translation completed" if result['success'] else "Translation failed")
        return result if result['success'] else None

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


    def check_translation_exit_command(self, text):
        """
        Check if the text contains a command to exit translation mode.

        Args:
            text (str): Text to check

        Returns:
            bool: True if exit command detected
        """
        return self.translation_handler.detect_translation_exit_phrase(text)

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

    def _get_language_code(self, language):
        """
        Convert language name to language code.

        Args:
            language (str): Language name or code

        Returns:
            str: Language code or None if invalid
        """
        # If it's already a code, return it
        if language in IO_CONFIG['SUPPORTED_LANGUAGES_CODES']:
            return language

        return IO_CONFIG['language_map'].get(language.lower(), None)

    def cleanup(self):
        """Clean up resources used by the LLM_Manager and its handlers."""
        self.logger.info("Cleaning up LLM_Manager resources")

        # Clean up translation resources
        if hasattr(self.translation_handler, 'release_translation_resources'):
            self.translation_handler.release_translation_resources()

        # Clear speech models
        with self._model_lock:
            self.speech_models.clear()

