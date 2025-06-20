# config.py

# Camera Handler Configuration
CAMERA_CONFIG = {
    # Camera resolution
    'CAPTURE_WIDTH': 1920,
    'CAPTURE_HEIGHT': 1080,

    # Image settings
    'IMAGE_QUALITY': 90,
    'SAVE_DIRECTORY': r"/tmp/AIAssistant/",
    'IMAGE_FILENAME': "captured_image.jpg",

    # Camera parameters
    'DEFAULT_EXPOSURE': 10000,
    'DEFAULT_GAIN': 1.0,
    'FRAME_RATE': 30,  # fps
    'FRAME_DURATION': 33333,  # microseconds (1/30 sec)

    # Camera controls
    'AWB_ENABLE': True,
    'AE_ENABLE': True,
    'DEFAULT_FOCUS_MODE': 'continuous',

    # Transform settings
    'HFLIP': 1,
    'VFLIP': 1,

    # Color format
    'COLOR_FORMAT': "RGB888"
}

# IO Manager Configuration
IO_CONFIG = {
    # Thread settings
    'CAMERA_THREAD_TIMEOUT': 0.5,
    'FRAME_INTERVAL': 0.03,  # 30fps

    # Audio settings
    'AUDIO_RECORD_TIMEOUT': 5,
    'WAKE_WORD': "hi david",

    # GUI window names
    'CAMERA_WINDOW_NAME': "camera window",
    'AI_WINDOW_NAME': "ai window",
    'User_INPUT_WINDOW_NAME': "user input window",

    'MAX_ATTEMPTS': 3,  # Max attempts for user input verification

    'MODE_KEYWORDS' : {
            "speech": ["speech", "voice", "audio", "speak", "one", "1"],
            "image": ["image", "picture", "photo", "text", "two", "2"],
            "both": ["both", "combined", "all", "three", "3"]
        },

    'COMMAND_KEYWORDS': {
        'start': ['start', 'begin', 'launch', 'activate', 'open'],
        'stop': ['stop', 'end', 'finish', 'quit'],
        'translate': ['translate', 'convert', 'change', 'interpret'],
        'exit': ['exit', 'quit', 'close', 'leave']
    },

}

# LLM Configuration
LLM_CONFIG = {

# Supported  modes mapping
    'PROMPTS_SUPPORTED' : {
        'translate': "tr",
        'extract': "ex",
    },

}

# OCR Configuration
OCR_CONFIG = {
    'DEFAULT_MODE': 'default',
    'OCR_MODES': {
        'default': '--oem 3 --psm 3',
        'accurate': '--oem 3 --psm 6'},

    'PREPROCESSING_LEVEL': 'medium',
    'SAVE_DIRECTORY': r"/tmp/AIAssistant/",
    'THRESH_VALUE': 150,
    'KERNEL_SIZE': (1, 1),
    'PROCESSED_FRAME_FILENAME': "processed_frame.jpg",
}

# LTD Configuration
LTD_CONFIG = {
    'TRANSLATION_MODELS_DIR': '/path/to/translation/models/',
    'MODEL_NAME': r'Helsinki-NLP/opus-mt-{source}-{target}',
    'DEFAULT_LANGUAGE': 'en',
    'MAX_LENGTH': 512,
    'BATCH_SIZE': 8,
    'MODELS_LOAD_TIMEOUT': 30,  # seconds
}

# Services Configuration
SERVICES_CONFIG = {
    # FuzzyWuzzy confidence_threshold
    'FUZZY_CONFIDENCE_THRESHOLD': 75,

    # Vosk model directory
    'RECOGNIZER_MODEL_PATH': r"/path/to/kaldi/model",
    # Recognizer model paths for different languages
    'RECOGNIZER_MODEL_NAME':{'en': r"vosk-model-small-en-us-0.15",
                            'ar': r"vosk-model-ar-mgb2-0.4",
                            'fr': r"vosk-model-small-fr-0.22"},

    #
    'SUPPORTED_LANGUAGES': ['en', 'ar', 'fr'],
    # Supported languages codes
    'SUPPORTED_LANG_CODES': {
        ["en", "ar", "fr"]
    },

    # Supported languages mapping
    'LANGUAGES_MAP': {
        'arabic': 'ar',
        'english': 'en',
        'french': 'fr',
    },
}