from PyQt5.QtCore import QThread, pyqtSignal
from utils.logging import Logger
import time

class WorkerThread(QThread):
    """Thread for background tasks with result signaling capability"""
    finished = pyqtSignal()
    result = pyqtSignal(object)

    def __init__(self, task_func, *args, task_name=None):
        """
        Initialize a WorkerThread with a task function and arguments.

        Args:
            task_func: The function to run in the separate thread
            *args: The arguments to pass to the function
            task_name: Optional name for logging purposes
        """
        super().__init__()
        self.task_func = task_func
        self.args = args
        self._is_running = True
        self.task_name = task_name or task_func.__name__
        self.logger = Logger()

    def run(self):
        """Run the task function in a separate thread."""
        if self._is_running:
            start_time = time.time()
            try:
                self.logger.info(f"Starting worker task: {self.task_name}")
                result = self.task_func(*self.args)
                self.result.emit(result)

                # Log performance data
                duration = time.time() - start_time
                self.logger.log_performance(f"worker_{self.task_name}", duration)
                self.logger.info(f"Completed worker task: {self.task_name}", duration=f"{duration:.3f}s")
            except Exception as e:
                self.logger.log_error_with_traceback(f"Error in worker task: {self.task_name}", e)
            finally:
                self.finished.emit()

    def stop(self):
        """Stop the thread safely."""
        self.logger.debug(f"Stopping worker thread: {self.task_name}")
        self._is_running = False
        self.quit()
        self.wait()
