vosk_languages = {
    "English": "en",
    "Arabic": "ar",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Russian": "ru",
    "Portuguese": "pt",
    "Chinese ": "cn",
    "Turkish": "tr",
    "Italian": "it",
    "Ukrainian": "uk",
    "Dutch": "nl",
    "Hindi": "hi",
    "Vietnamese": "vi",
    "Korean": "ko",
    "Japanese": "ja",
    "Polish": "pl"
}

tesseract_languages = {
    "English": "eng",
    "Arabic": "ara",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Russian": "rus",
    "Portuguese": "por",
    "Chinese": "chi_sim",
    "Turkish": "tur",
    "Italian": "ita",
    "Ukrainian": "ukr",
    "Dutch": "nld",
    "Hindi": "hin",
    "Vietnamese": "vie",
    "Korean": "kor",
    "Japanese": "jpn",
    "Polish": "pol"
}

vosk_to_tesseract = {
    "en": "eng",
    "ar": "ara",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "ru": "rus",
    "pt": "por",
    "cn": "chi_sim",
    "tr": "tur",
    "it": "ita",
    "uk": "ukr",
    "nl": "nld",
    "hi": "hin",
    "vi": "vie",
    "ko": "kor",
    "ja": "jpn",
    "pl": "pol"
}

# IMPORTANT: Update these paths to be correct for your system if they are not already.
# Consider using relative paths or environment variables for better portability.
vosk_model_paths = {
    "en": r"vosk-model-en-us-daanzu-20200905-lgraph",
    "ar": r"vosk-model-ar-mgb2-0.4",
    "es": r"vosk-model-small-es-0.42",
    "fr": r"vosk-model-small-fr-0.22",
    "de": r"vosk-model-small-de-0.15",
    "ru": r"vosk-model-small-ru-0.22",
    "pt": r"vosk-model-small-pt-0.3",
    "cn": r"vosk-model-small-cn-0.22",
    "tr": r"vosk-model-small-tr-0.3",
    "it": r"vosk-model-small-it-0.22",
    "uk": r"vosk-model-small-uk-v3-nano",
    "nl": r"vosk-model-small-nl-0.22",
    "hi": r"vosk-model-small-hi-0.22",
    "vi": r"vosk-model-small-vn-0.4", # Assuming this path exists, adjust if needed
    "ko": r"vosk-model-small-ko-0.22",
    "ja": r"vosk-model-small-ja-0.22",
    "pl": r"vosk-model-small-pl-0.22"
}

# Path to Tesseract executable
# Update this if your Tesseract installation is in a different location.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# API Endpoints
API_BASE_URL = "http://127.0.0.1:8000/api/"
API_CHAT_ENDPOINT = f"{API_BASE_URL}chat/"
API_TRANSCRIBE_ENDPOINT = f"{API_BASE_URL}transcribe-audio/"

# Camera/UI settings
LABELS_WIDTH_PERCENT = 0.3
CAMERA_WIDTH_PERCENT = 1 - LABELS_WIDTH_PERCENT
GESTURE_COOLDOWN = 1
DEFAULT_USERNAME = "User" # Default username if not specified