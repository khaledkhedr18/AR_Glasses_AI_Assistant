from fuzzywuzzy import fuzz
from utils.logging import Logger


class LTD_Handler:
    """
    Handles language and tool detection without direct dependencies on other handlers.
    """

    def __init__(self):
        """Initialize the LTD_Handler."""
        self.logger = Logger()
        self.logger.info("Initializing LTD_Handler")

        self.supported_languages = {
            "english": "en",
            "arabic": "ar",
            "french": "fr"
        }

    def detect_language_from_text(self, text, confidence_threshold=70):
        """
        Detect language from text using fuzzy matching.

        Args:
            text (str): Text to analyze
            confidence_threshold (int): Minimum score to consider match valid

        Returns:
            tuple: (language_code, confidence) or (None, 0) if no match
        """
        if not text or not text.strip():
            return None, 0

        # Calculate fuzzy match scores for each supported language
        scores = {}
        for language_name, language_code in self.supported_languages.items():
            scores[language_code] = fuzz.partial_ratio(text.lower(), language_name.lower())

        # Find the best match
        best_match = max(scores, key=scores.get)
        best_score = scores[best_match]

        if best_score > confidence_threshold:
            self.logger.info(f"Language detected: {best_match} (confidence: {best_score}%)")
            return best_match, best_score
        else:
            return None, 0

    def detect_tool_from_text(self, text):
        """
        Detect whether the user wants speech or image processing.

        Args:
            text (str): User input text

        Returns:
            str or None: "speech", "image", or None if undetermined
        """
        if not text:
            return None

        text_lower = text.lower()

        # Check for speech indicators
        if any(word in text_lower for word in ["speech", "voice", "audio", "speak", "talk", "one", "1"]):
            return "speech"

        # Check for image indicators
        elif any(word in text_lower for word in ["image", "picture", "photo", "text", "read", "scan", "two", "2"]):
            return "image"

        # Unable to determine
        return None

    def get_language_code(self, language_name):
        """
        Get language code from language name.

        Args:
            language_name (str): Language name

        Returns:
            str or None: Language code or None if not supported
        """
        if not language_name:
            return None

        # Check if it's already a code
        if language_name.lower() in ["en", "ar", "fr"]:
            return language_name.lower()

        # Try to match with supported languages
        return self.supported_languages.get(language_name.lower(), None)
