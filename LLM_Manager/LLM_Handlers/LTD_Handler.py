from fuzzywuzzy import fuzz
from IO_Manager.IO_Handlers.Audio_Handler import AudioHandler


class LTD_Handler(QObject):
    """
    This class handles the language and the tool detection
    """
    def __init__(self, audio_handler=None):
        """
        Initialize the DL_Handler.

        Args:
            audio_handler (AudioHandler, optional): An instance of AudioHandler.
                If None, a new instance will be created.
        """
        super().__init__()
        self.audio_handler = audio_handler or AudioHandler()
        self.supported_languages = {
            "english": "en",
            "arabic": "ar",
            "french": "fr"
        }

    def get_lang1(self, model):
        """
        Prompts the user to specify the source language for translation.

        The function asks the user to specify the source language by speaking
        either "English", "Arabic", or "French". It uses fuzzy matching to interpret
        the user's response and returns the corresponding language code ("en", "ar",
        "fr") if a match is found with a confidence score above 70.

        Args:
            model: The speech recognition model used to capture user input.

        Returns:
            str or None: The language code for the source language if a match is
            found, otherwise None.
        """
        self.audio_handler.speak("What is the source language? English, Arabic, or French?")
        print("What is the source language? (English/Arabic/French)")

        return self._get_language_selection(model)

    def get_lang2(self, model):
        """
        Prompts the user to specify the target language for translation.

        Uses the same logic as get_lang1 but for the target language.

        Args:
            model: The speech recognition model used to capture user input.

        Returns:
            str or None: The language code for the target language if a match is
            found, otherwise None.
        """
        self.audio_handler.speak("What is the target language? English, Arabic, or French?")
        print("What is the target language? (English/Arabic/French)")

        return self._get_language_selection(model)

    def _get_language_selection(self, model, max_attempts=5, confidence_threshold=70):
        """
        Helper method to handle language selection with fuzzy matching.

        Args:
            model: The speech recognition model to use
            max_attempts (int): Maximum number of attempts to recognize language
            confidence_threshold (int): Minimum score to consider a match valid

        Returns:
            str or None: Language code if match found, otherwise None
        """
        for attempt in range(max_attempts):
            user_input = self.audio_handler.get_audio(model)
            if not user_input or not user_input.strip():
                self.audio_handler.speak("I didn't hear you. Please try again.")
                continue

            print(f"Detected speech: '{user_input}'")

            # Calculate fuzzy match scores for each supported language
            scores = {}
            for language_name, language_code in self.supported_languages.items():
                scores[language_code] = fuzz.partial_ratio(user_input.lower(), language_name.lower())

            # Find the best match
            best_match = max(scores, key=scores.get)
            best_score = scores[best_match]

            if best_score > confidence_threshold:
                print(f"Language detected: {best_match} (confidence: {best_score}%)")
                return best_match
            else:
                self.audio_handler.speak("I'm not sure which language you mean. Please say English, Arabic, or French.")

        self.audio_handler.speak("I couldn't understand your language selection. Please try again later.")
        return None

    def tool_detection(self, model):
        """
        Determines whether the user wants to use speech or image translation.

        Args:
            model: The speech recognition model to use

        Returns:
            str or None: "speech", "image", or None if selection couldn't be determined
        """
        self.audio_handler.speak("Do you want to translate speech or image?")
        print("Speech or image input?")

        max_attempts = 3
        for attempt in range(max_attempts):
            user_input = self.audio_handler.get_audio(model)
            if not user_input:
                continue

            user_input = user_input.lower()

            # Check for speech indicators
            if any(word in user_input for word in ["speech", "voice", "audio", "speak", "one", "1"]):
                return "speech"

            # Check for image indicators
            elif any(word in user_input for word in ["image", "picture", "photo", "text", "two", "2"]):
                return "image"

            # If we get here, the input wasn't clear
            self.audio_handler.speak("Please say 'speech' or 'image'.")

        self.audio_handler.speak("I couldn't determine your selection. Please try again.")
        return None
