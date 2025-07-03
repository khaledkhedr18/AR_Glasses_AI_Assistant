"""
Configuration settings for AR Glasses AI Assistant
"""

# GUI Widget Configuration
OVERLAY_WIDGET_CONFIGS = {
    "status": {
        "style": """
            QLabel {
                color: white;
                background-color: rgba(0,0,0,200);
                border-radius: 5px;
                padding: 5px;
                font: bold 14px;
            }
        """,
        "alignment": "center",
        "size": (200, 40),
        "position": "center_top",  # Special value we'll handle in Qt_Handler
        "default_text": "Status: INITIALIZING"
    },
    "user_speech": {
        "style": """
            QLabel {
                color: white;
                background-color: rgba(0,0,0,200);
                border-radius: 5px;
                padding: 8px;
                font: 12px;
            }
        """,
        "size": (300, 100),
        "position": "bottom_right",  # Special value
        "word_wrap": True,
        "default_text": "You: "
    },
    "ai_response": {
        "style": """
            QLabel {
                color: white;
                background-color: rgba(0,0,0,200);
                border-radius: 5px;
                padding: 8px;
                font: 12px;
            }
        """,
        "size": (300, 100),
        "position": "bottom_left",  # Special value
        "word_wrap": True,
        "default_text": "Assistant: Ready"
    }
}

# Camera Handler Configuration
CAMERA_CONFIG = {
    # Camera resolution
    'RESOLUTION': {
        'WIDTH': 1920,
        'HEIGHT': 1080
    },

    # Image settings
    'IMAGE': {
        'QUALITY': 90,
        'SAVE_DIRECTORY': r"./tmp/AIAssistant/",
        'FILENAME': "captured_image.jpg"
    },

    # Camera parameters
    'PARAMETERS': {
        'EXPOSURE': 10000,
        'GAIN': 1.0,
        'FRAME_RATE': 30,
        'FRAME_DURATION': 33333
    },

    # Camera controls
    'CONTROLS': {
        'AWB_ENABLE': True,
        'AE_ENABLE': True,
        'FOCUS_MODE': 'continuous',
        'HFLIP': 1,
        'VFLIP': 1,
        'COLOR_FORMAT': "RGB888"
    }
}

# IO Manager Configuration
IO_CONFIG = {
    # Thread settings
    'TIMING': {
        'CAMERA_THREAD_TIMEOUT': 0.5,
        'FRAME_INTERVAL': 0.03,  # 30fps
        'AUDIO_RECORD_TIMEOUT': 3
    },
    'AUDIO': {
        'SAVE_DIRECTORY': r"./tmp/audio_files/",
        'MAX_DURATION': 10
    }
    ,
    # Interface settings
    'INTERFACE': {
        'MAX_ATTEMPTS': 3,
        'WINDOWS': {
            'CAMERA': "camera window",
            'AI': "ai window",
            'USER_INPUT': "user input window"
        }
    },

    # User Commands
    'USER_COMMANDS' : {
        # Keywords for commands
        'KEYWORDS': {
            'MODE': {
                'SPEECH': ["speech", "voice", "audio", "speak", "one", "1"],
                'IMAGE': ["image", "picture", "photo", "text", "two", "2"],
                'BOTH': ["both", "combined", "all", "three", "3"]
        },
        'COMMANDS': {
            'START': ['start', 'begin', 'launch', 'activate', 'open'],
            'STOP': ['stop', 'end', 'finish', 'quit'],
            'TRANSLATE': ['translate', 'convert', 'change', 'interpret'],
            'EXIT': ["quit", "exit", "stop", "close", "terminate", "shutdown"]
        },

        "WAKE_WORD": {
            "hey david", "hi david", "hello david", "david", "assistant"
        },
        "HIDE_WIDGET_COMMAND": {
            "hide", "minimize", "collapse", "conceal", "shrink"
        },
        "SHOW_WIDGET_COMMAND": {
            "show", "expand", "reveal", "display", "enlarge"
        }
    },

    # Language settings
    'SUPPORTED_LANGUAGES': {
        'arabic': 'ar',
        'english': 'en',
        'french': 'fr',
    }
}
}


# OCR Configuration
OCR_CONFIG = {
    'LANGUAGE_MAPPING': {
        # Two-letter to three-letter code mapping
        'en': 'eng',
        'ar': 'ara',
        'fr': 'fra',
        # Add more languages as needed
    },
    'TESSERACT_PATHS': {
        'cmd': '/usr/bin/tesseract',
        'data': '/usr/share/tesseract-ocr/5/tessdata/'
    },
    'MODES': {
        'DEFAULT': '--oem 3 --psm 3',
        'ACCURATE': '--oem 3 --psm 6'
    },

    'PROCESSING': {
        'LEVEL': 'medium',
        'THRESH_VALUE': 150,
        'KERNEL_SIZE': (1, 1),
        'DENOISE_H': 10,
        'MAX_VALUE': 255
    },

    'STORAGE': {
        'SAVE_DIRECTORY': r"/tmp/AIAssistant/",
        'PROCESSED_FRAME_FILENAME': "processed_frame.jpg"
    }
}

# Machine Learning Configuration
ML_CONFIG = {
    'TRANSLATION': {
        'TRANSLATION_MODELS_DIR': './models/translation',
        'MODEL_NAMES': {
            'en-ar': 'Helsinki-NLP/opus-mt-en-ar',  # English to Arabic
            'fr-en': 'Helsinki-NLP/opus-mt-fr-en',  # French to English
            'ar-en': 'Helsinki-NLP/opus-mt-ar-en',  # Arabic to English
        },
        'MODELS_LOAD_TIMEOUT': 60,  # Increased timeout for Pi
        'MODELS_CACHE_DIR': './models/translation/cache',
        'SUPPORTED_SPEECH_MODELS': ['en'],
        'SUPPORTED_TRANSLATION_PAIRS': [
            ('en', 'ar'),
            ('en', 'fr'),
            ('ar', 'en'),
        ],
        'MAX_WORKERS': 2,
        'USE_LOW_MEMORY': False,
    }
}

# Services Configuration
SERVICES_CONFIG = {
    'LANGUAGES': {
        'SUPPORTED': ['en', 'ar', 'fr'],
        'MAPPING': {
            'english': 'en',
            'arabic': 'ar',
            'french': 'fr'
        }
    },

    'RECOGNITION': {
        'VOSK_MODEL_DIR': 'models/vosk',
        'VOSK_MODELS': {
            'en': 'vosk-model-small-en-us-0.15'
        },
        'FUZZY_CONFIDENCE_THRESHOLD': 75
    }
}

# LLM Configuration
LLM_CONFIG = {
    'PROMPTS': {
        'SUPPORTED': {
            'translate': "tr",
            'extract': "ex"
        }
    }
}

# Network Manager Configuration
NETWORK_CONFIG = {
    # Server connection settings
    'SERVER_IP': '192.168.1.108',
    'SERVER_PORT': 4040,
    'SOCKET_TIMEOUT': 5,  # seconds
    'MAX_RETRIES': 3,
    'RETRY_DELAY': 1,  # seconds
    'CONNECTION_CHECK_INTERVAL': 10,  # seconds

    # Request settings
    'REQUEST_TIMEOUT': 30,  # seconds
    'MAX_PAYLOAD_SIZE': 10 * 1024 * 1024,  # 10MB max payload size
    'CHUNK_SIZE': 4096,  # bytes

    # Supported data types
    'DATA_TYPES': ['text', 'image', 'text_and_image'],

    # Response settings
    'RESPONSE_TIMEOUT': 30,  # seconds

    # Status checks
    'STATUS_CHECK_URLS': ['https://www.google.com', 'https://www.cloudflare.com'],
    'STATUS_CHECK_PORT': 80,
    'STATUS_CHECK_TIMEOUT': 2.0,  # seconds

    # Threading
    'THREAD_JOIN_TIMEOUT': 1.0,  # seconds

    # Debug options
    'VERBOSE_LOGGING': False,
    'LOG_PAYLOADS': False,  # Be careful with this in production!
}

# User Preferences
user_config = {
    "source_language": "english",
    "target_language": "arabic",
    "tool_detection": "image",
    "if_online": True,
}

# Application Configuration
APP_CONFIG = {
    'DISPLAY': {
        'FULLSCREEN': True,
        'RESOLUTION': (800, 600),
        'THEME': 'dark',
        'OPACITY': 0.9
    },
    'APPLICATION': {
        'NAME': 'AR Glasses Assistant',
        'VERSION': '0.1.0',
        'AUTO_START': True
    },
    'PERFORMANCE': {
        'LOW_RESOURCE_MODE': False,
        'LOG_PERFORMANCE': True
    }
}
