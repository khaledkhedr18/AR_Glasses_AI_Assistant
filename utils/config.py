"""
Configuration settings for Qt overlay widgets
"""

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
        "position": (300, 20),
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
        "position": (480, 360),
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
        "position": (20, 360),
        "word_wrap": True,
        "default_text": "Assistant: Ready"
    }
}


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
    'WAKE_WORD': "hey david",

    # GUI window names
    'CAMERA_WINDOW_NAME': "camera window",
    'AI_WINDOW_NAME': "ai window",
    'User_INPUT_WINDOW_NAME': "user input window",

    # Language settings
    'SUPPORTED_LANGUAGES': {
        'arabic': 'ar',
        'english': 'en',
        'french': 'fr',
        'spanish': 'es',
        'german': 'de'
    },

    'COMMAND_KEYWORDS': {
        'start': ['start', 'begin', 'launch', 'activate', 'open'],
        'stop': ['stop', 'end', 'finish', 'quit'],
        'translate': ['translate', 'convert', 'change', 'interpret'],
        'exit': ['exit', 'quit', 'close', 'leave']
    },

    # Models Paths
    'RECOGNIZER_MODEL_PATH': r"/path/to/kaldi/model",
}

# Network Manager Configuration
NETWORK_CONFIG = {
    # Server connection settings
    'SERVER_IP': '192.168.1.65',
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

user_config = {
    "source_language": "english",
    "target_language": "arabic",
    "tool_detection": "image",
    "if_online": True,
}





user_commands = {
    "exit_command": {
        "quit", "exit", "stop", "close", "terminate", "shutdown"
    },
    "wake_word": {
        "hey david", "hi david", "hello david", "david", "assistant"
    },
    "hide_widget_command": {
        "hide", "minimize", "collapse", "conceal", "shrink"
    },
    "show_widget_command": {
        "show", "expand", "reveal", "display", "enlarge"
    },
}


