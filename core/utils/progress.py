from typing import List, Callable
from PyQt5.QtCore import QObject, pyqtSignal
from core.utils.logging import Logger

class ProgressMonitor(QObject):
    """Monitor for tracking operation progress"""

    progress_updated = pyqtSignal(int, str)  # progress percentage, status message

    def __init__(self):
        super().__init__()
        self._progress = 0
        self._status = ""
        self._observers: List[Callable[[int, str], None]] = []
        self.logger = Logger()

    @property
    def progress(self) -> int:
        """Get current progress percentage"""
        return self._progress

    @property
    def status(self) -> str:
        """Get current status message"""
        return self._status

    def update(self, progress: int, status: str):
        """Update progress and status"""
        self._progress = max(0, min(100, progress))  # Clamp between 0-100
        self._status = status
        self.progress_updated.emit(self._progress, self._status)
        self.notify_observers()
        self.logger.debug(f"Progress: {self._progress}% - {self._status}")

    def add_observer(self, observer: Callable[[int, str], None]):
        """Add progress observer"""
        if observer not in self._observers:
            self._observers.append(observer)

    def remove_observer(self, observer: Callable[[int, str], None]):
        """Remove progress observer"""
        if observer in self._observers:
            self._observers.remove(observer)

    def notify_observers(self):
        """Notify all observers of progress update"""
        for observer in self._observers:
            try:
                observer(self._progress, self._status)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")

    def reset(self):
        """Reset progress and status"""
        self.update(0, "")
