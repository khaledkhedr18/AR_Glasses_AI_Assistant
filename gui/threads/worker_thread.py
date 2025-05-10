from PyQt5.QtCore import QThread, pyqtSignal
from typing import Callable, Any, Tuple

class WorkerThread(QThread):
    """Thread for running background tasks"""

    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, task_func: Callable, *args):
        """
        Initialize worker thread with task function and arguments

        Args:
            task_func: Function to run in the thread
            *args: Arguments to pass to the function
        """
        super().__init__()
        self.task_func = task_func
        self.args = args
        self._is_running = True

    def run(self):
        """Run the task function"""
        if not self._is_running:
            return

        try:
            result = self.task_func(*self.args)
            self.result.emit(result)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        """Stop the thread"""
        self._is_running = False
        self.quit()
        self.wait()
