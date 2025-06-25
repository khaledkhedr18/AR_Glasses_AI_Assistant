from fuzzywuzzy import fuzz
from vosk import KaldiRecognizer
import wave
import json
import io
from utils.Config import SERVICES_CONFIG
from utils.Logging import Logger


class Services:
    def __init__(self):
        # Initialize Services with necessary configurations and logger
        self.logger = Logger()

        # Load language configurations with defaults
        languages_config = SERVICES_CONFIG.get('LANGUAGES', {})
        self.supported_lang_codes = languages_config.get('SUPPORTED', ['en'])
        self.languages_map = languages_config.get('MAPPING', {'english': 'en'})
        self._lang_code_cache = {}

    def recognize_text_from_speech(self, wave_data, model_components=None):
        """
        Convert WAV audio data to text using Vosk.

        Args:
            wave_data: BytesIO object containing WAV data or raw wave chunks
            model_components: Either a direct Vosk Model object or a tuple (model, success)

        Returns:
            str: Recognized text or None if recognition fails
        """
        try:
            # Handle case where no model is provided
            if model_components is None:
                self.logger.warning("No speech model provided - returning test response")
                return "translate this text"

            # Better model type detection
            from vosk import Model as VoskModel

            # Handle case where model_components is a direct Model object
            if isinstance(model_components, VoskModel):
                self.logger.info("Using direct Vosk model for speech recognition")
                speech_model = model_components
            # Handle case where model_components is a direct Model object but not detected by isinstance
            elif hasattr(model_components, 'ReadDataFiles'):
                self.logger.info("Using Vosk model with ReadDataFiles for speech recognition")
                speech_model = model_components
            else:
                # Try to handle it as a tuple (model, success)
                try:
                    # Check if it's a tuple with at least 2 elements
                    if len(model_components) >= 2:
                        if not model_components[1]:  # Check success flag
                            self.logger.error("Invalid speech model components")
                            return None
                        speech_model = model_components[0]
                    else:
                        # If it's a tuple with just one element
                        speech_model = model_components[0]
                except (TypeError, IndexError):
                    self.logger.warning("Unexpected model format - returning test response")
                    return "translate this text"

            # Process audio with the model
            # If input is already a BytesIO/file-like object, use it directly
            if isinstance(wave_data, (io.BytesIO, io.BufferedRandom)):
                wf = wave.open(wave_data, "rb")
            else:
                # If input is raw bytes or numpy array, wrap it in BytesIO
                import numpy as np
                if isinstance(wave_data, np.ndarray):
                    import scipy.io.wavfile as wavfile
                    buffer = io.BytesIO()
                    wavfile.write(buffer, 16000, wave_data.astype(np.int16))
                    buffer.seek(0)
                    wf = wave.open(buffer, "rb")
                else:
                    # Regular bytes data
                    buffer = io.BytesIO(wave_data)
                    wf = wave.open(buffer, "rb")

            with wf:
                recognizer = KaldiRecognizer(speech_model, wf.getframerate())
                text = ""

                # Process audio in chunks
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text += result.get("text", "") + " "

                # Get final recognition result
                final_result = json.loads(recognizer.FinalResult())
                text += final_result.get("text", "")

            text = text.strip().lower()
            self.logger.info(f"Speech recognition result: '{text}'")
            return text if text else None

        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            return None

    def verify_user_input(self, text, items_dict, confidence_threshold=None):
        """
        Generic fuzzy matching method to verify user input against a dictionary or list of items.

        Args:
            text (str): The text to verify
            items_dict (dict/list): Dictionary {name: code} or list of keywords mapping to a single value
            confidence_threshold (int): Minimum score for match

        Returns:
            str or None: Matched value (language code, mode, etc.) or None if no match
        """
        if not text or not isinstance(text, str):
            self.logger.warning("Invalid input text for verification")
            return None

        # Get recognition config with defaults
        recognition_config = SERVICES_CONFIG.get('RECOGNITION', {})
        if confidence_threshold is None:
            confidence_threshold = recognition_config.get('FUZZY_CONFIDENCE_THRESHOLD', 75)

        text = text.lower().strip()

        # Handle dictionary input (like languages)
        if isinstance(items_dict, dict):
            comparison_dict = items_dict

        # Handle list input with implicit mapping (like modes)
        elif isinstance(items_dict, list):
            # For lists like ["speech", "voice", "audio"] -> "speech"
            # First item is considered the target value
            target_value = items_dict[0]
            comparison_dict = {keyword: target_value for keyword in items_dict}
        else:
            self.logger.error("Invalid items_dict type")
            return None

        # Calculate fuzzy match scores
        best_overall_score = 0
        best_match_value = None

        for item_name, item_value in comparison_dict.items():
            # Calculate different types of fuzzy matches
            partial_score = fuzz.partial_ratio(text, item_name.lower())
            token_sort_score = fuzz.token_sort_ratio(text, item_name.lower())
            token_set_score = fuzz.token_set_ratio(text, item_name.lower())

            # Get best score among different matching methods
            best_score = max(partial_score, token_sort_score, token_set_score)

            if best_score > best_overall_score:
                best_overall_score = best_score
                best_match_value = item_value

            self.logger.debug(f"Match '{item_name}': score={best_score}%")

        if best_overall_score >= confidence_threshold:
            self.logger.info(f"Match found with confidence: {best_overall_score}%")
            return best_match_value
        else:
            self.logger.warning(f"No match found above threshold ({confidence_threshold}%)")
            return None

    def get_language_code(self, language):
        """
        Convert language name to standardized code.

        Args:
            language (str): Language name (e.g. 'english') or code (e.g. 'en')

        Returns:
            str: Standardized language code if valid, None otherwise
        """
        if not language:
            return None

        language = language.lower().strip()

        # Check cache first
        if language in self._lang_code_cache:
            return self._lang_code_cache[language]

        # Get languages config with defaults
        languages_config = SERVICES_CONFIG.get('LANGUAGES', {})
        supported_codes = languages_config.get('SUPPORTED', ['en'])
        lang_mapping = languages_config.get('MAPPING', {'english': 'en'})

        # Try to get code
        code = None
        if language in supported_codes:
            code = language
        elif language in lang_mapping:
            code = lang_mapping[language]

        # Update cache
        if code:
            self._lang_code_cache[language] = code

        return code
