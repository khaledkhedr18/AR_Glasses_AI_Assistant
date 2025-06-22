import threading
import time
from utils.Logging import Logger

class BaseWorkerThread:
    """Base class for worker threads with callback capability"""

    def __init__(self, task_func, *args, on_result=None, on_finished=None, task_name=None):
        """
        Initialize a worker thread with a task function and arguments.

        Args:
            task_func: The function to run in the separate thread
            *args: The arguments to pass to the function
            on_result: Callback function to receive the result (optional)
            on_finished: Callback function called when thread finishes (optional)
            task_name: Optional name for logging purposes
        """
        self.task_func = task_func
        self.args = args
        self._is_running = True
        self.task_name = task_name or task_func.__name__
        self.on_result = on_result
        self.on_finished = on_finished
        self.logger = Logger()

    def _run_task(self):
        """Run the task and handle callbacks"""
        start_time = time.time()
        try:
            self.logger.info(f"Starting worker task: {self.task_name}")
            result = self.task_func(*self.args)

            # Send result through callback if provided
            if self.on_result and self._is_running:
                self.on_result(result)

            # Log performance data
            duration = time.time() - start_time
            self.logger.log_performance(f"worker_{self.task_name}", duration)
            self.logger.info(f"Completed worker task: {self.task_name}", duration=f"{duration:.3f}s")
            return result
        except Exception as e:
            self.logger.log_error_with_traceback(f"Error in worker task: {self.task_name}", e)
            return None
        finally:
            if self.on_finished and self._is_running:
                self.on_finished()

    def start(self):
        """Start the worker thread - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement start()")

    def stop(self):
        """Stop the worker thread - to be implemented by subclasses"""
        self._is_running = False


class ThreadingWorker(BaseWorkerThread, threading.Thread):
    """Standard Python threading implementation of WorkerThread"""

    def __init__(self, task_func, *args, on_result=None, on_finished=None, task_name=None):
        BaseWorkerThread.__init__(self, task_func, *args, on_result=on_result,
                                 on_finished=on_finished, task_name=task_name)
        threading.Thread.__init__(self)
        self.daemon = True  # Thread will exit when main program exits

    def run(self):
        """Run the task function in a separate thread."""
        if self._is_running:
            self._run_task()

    def start(self):
        """Start the thread"""
        threading.Thread.start(self)
        return self


# Factory function to create the appropriate worker type
def create_worker(task_func, *args, on_result=None, on_finished=None, task_name=None, worker_type="thread"):
    """
    Create an appropriate worker based on the specified type

    Args:
        task_func: Function to run in the thread
        *args: Arguments for the function
        on_result: Callback for when results are available
        on_finished: Callback for when thread completes
        task_name: Name for logging purposes
        worker_type: Type of worker ("thread" for standard threading)

    Returns:
        A worker instance appropriate for the environment
    """
    # Try to import Qt if it's requested
    if worker_type == "qt":
        try:
            from PyQt5.QtCore import QThread, pyqtSignal

            # Create Qt implementation dynamically
            class QtWorker(BaseWorkerThread, QThread):
                result_signal = pyqtSignal(object)
                finished_signal = pyqtSignal()

                def __init__(self, task_func, *args, on_result=None, on_finished=None, task_name=None):
                    BaseWorkerThread.__init__(self, task_func, *args, on_result=on_result,
                                            on_finished=on_finished, task_name=task_name)
                    QThread.__init__(self)

                    # Connect signals to callbacks if provided
                    if on_result:
                        self.result_signal.connect(on_result)
                    if on_finished:
                        self.finished_signal.connect(on_finished)

                def run(self):
                    """Run the task function in a separate thread."""
                    if self._is_running:
                        result = self._run_task()
                        if self._is_running:
                            self.result_signal.emit(result)
                            self.finished_signal.emit()

                def stop(self):
                    """Stop the thread safely."""
                    self.logger.debug(f"Stopping Qt worker thread: {self.task_name}")
                    self._is_running = False
                    self.quit()
                    self.wait()

            return QtWorker(task_func, *args, on_result=on_result,
                          on_finished=on_finished, task_name=task_name)
        except ImportError:
            # Fall back to standard threading if PyQt not available
            pass

    # Default to standard threading implementation
    return ThreadingWorker(task_func, *args, on_result=on_result,
                         on_finished=on_finished, task_name=task_name)
