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
