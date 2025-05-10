from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QFont

class OverlayLabel(QLabel):
    """Base class for overlay labels"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0,0,0,200);
                border-radius: 5px;
                padding: 8px;
                font: 12px;
            }
        """)
        self.setWordWrap(True)

class StatusLabel(OverlayLabel):
    """Label for displaying connection status"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(200, 40)
        self.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0,0,0,200);
                border-radius: 5px;
                padding: 5px;
                font: bold 14px;
            }
        """)

class SpeechLabel(OverlayLabel):
    """Label for displaying speech text"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(300, 100)
        self.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0,0,0,200);
                border-radius: 5px;
                padding: 8px;
                font: 12px;
            }
        """)
