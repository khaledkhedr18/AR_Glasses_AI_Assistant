import os
import sys
import threading
from utils.logging import Logger
from LLM_Manager.LLM_Handlers.Translation_Handler import Translation_Handler
from LLM_Manager.LLM_Handlers.OCR_Handler import OCR_Handler
from LLM_Manager.LLM_Handlers.LTD_Handler import LTD_Handler

class LLM_Manager:
    """
    Central manager for all language model operations.

    This class acts as a communication hub for:
    1. LLM handlers to communicate with each other indirectly
    2. External managers (like IO_Manager) to access language model functionality
    """

    def __init__(self, config=None, io_manager=None):
        """
        Initialize the LLM_Manager with all required handlers.

        Args:
            config (dict, optional): Configuration dictionary
            io_manager: Reference to IO_Manager for interface operations
        """
        self.config = config or {}
        self.io_manager = io_manager  # Will be set by main application
        self.logger = Logger()
        self.logger.info("Initializing LLM_Manager")

        # Initialize handlers without direct references to each other
        self.translation_handler = Translation_Handler()
        self.ocr_handler = OCR_Handler()
        self.ltd_handler = LTD_Handler()

        # Cache for speech models
        self.speech_models = {}
        self._model_lock = threading.Lock()

        # Default language settings
        self.default_source_lang = self.config.get('source_language', 'english')
        self.default_target_lang = self.config.get('target_language', 'arabic')

        self.logger.info("LLM_Manager initialized successfully")

    # ======== EXTERNAL INTERFACE METHODS ========

    def set_io_manager(self, io_manager):
        """Set the IO_Manager reference for interface operations."""
        self.io_manager = io_manager

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

    def translate_text(self, text, source_lang=None, target_lang=None):
        """
        Translate text between languages.

        Args:
            text (str): Text to translate
            source_lang (str, optional): Source language code or name
            target_lang (str, optional): Target language code or name

        Returns:
            str: Translated text or error message
        """
        self.logger.info(f"Translating text: '{text[:30]}{'...' if len(text) > 30 else ''}'")

        # Use defaults if languages not specified
        source_lang = source_lang or self.default_source_lang
        target_lang = target_lang or self.default_target_lang

        # Convert language names to codes if needed
        source_code = self._get_language_code(source_lang)
        target_code = self._get_language_code(target_lang)

        if not source_code or not target_code:
            self.logger.error(f"Invalid language specified: {source_lang} or {target_lang}")
            return None

        # Use translation handler
        result = self.translation_handler.translate_with_best_available_method(
            text, source_code, target_code
        )

        if result:
            return result
        else:
            self.logger.error("Translation failed")
            return None

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
        if language in ["en", "ar", "fr"]:
            return language

        # If it's a name, convert it
        language_map = {
            "english": "en",
            "arabic": "ar",
            "french": "fr"
        }

        return language_map.get(language.lower(), None)

    def cleanup(self):
        """Clean up resources used by the LLM_Manager and its handlers."""
        self.logger.info("Cleaning up LLM_Manager resources")

        # Clean up translation resources
        if hasattr(self.translation_handler, 'release_translation_resources'):
            self.translation_handler.release_translation_resources()

        # Clear speech models
        with self._model_lock:
            self.speech_models.clear()

