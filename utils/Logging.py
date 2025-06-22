import logging

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import threading

class Logger:
    """Enhanced application logger with performance monitoring and structured logging"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Logger, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize logger with enhanced features"""
        self.logger = logging.getLogger('AR_Assistant')
        self.logger.setLevel(logging.INFO)

        # Performance tracking
        self._performance_data: Dict[str, Any] = {}
        self._timing_lock = threading.Lock()

        # Create logs directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Enhanced file handler with rotation
        self._setup_file_handler()
        self._setup_console_handler()

        # Custom formatter with structured data
        self._setup_formatter()

    def _setup_file_handler(self):
        """Setup file handler with rotation"""
        from logging.handlers import RotatingFileHandler

        log_file = self.log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        self.file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        self.file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(self.file_handler)

    def _setup_console_handler(self):
        """Setup enhanced console handler"""
        self.console_handler = logging.StreamHandler(sys.stdout)
        self.console_handler.setLevel(logging.INFO)
        self.logger.addHandler(self.console_handler)

    def _setup_formatter(self):
        """Setup enhanced formatter"""
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)

    def info(self, message: str, **kwargs):
        """Enhanced info logging with structured data"""
        structured_msg = self._format_message(message, kwargs)
        self.logger.info(structured_msg)

    def error(self, message: str, **kwargs):
        """Enhanced error logging with structured data"""
        structured_msg = self._format_message(message, kwargs)
        self.logger.error(structured_msg)

    def warning(self, message: str, **kwargs):
        """Enhanced warning logging with structured data"""
        structured_msg = self._format_message(message, kwargs)
        self.logger.warning(structured_msg)

    def debug(self, message: str, **kwargs):
        """Enhanced debug logging with structured data"""
        structured_msg = self._format_message(message, kwargs)
        self.logger.debug(structured_msg)

    def critical(self, message: str, **kwargs):
        """Critical logging for severe errors"""
        structured_msg = self._format_message(message, kwargs)
        self.logger.critical(structured_msg)

    def _format_message(self, message: str, kwargs: Dict[str, Any]) -> str:
        """Format message with structured data"""
        if kwargs:
            extras = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            return f"{message} | {extras}"
        return message

    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics"""
        with self._timing_lock:
            if operation not in self._performance_data:
                self._performance_data[operation] = {
                    'count': 0,
                    'total_time': 0.0,
                    'min_time': float('inf'),
                    'max_time': 0.0
                }

            data = self._performance_data[operation]
            data['count'] += 1
            data['total_time'] += duration
            data['min_time'] = min(data['min_time'], duration)
            data['max_time'] = max(data['max_time'], duration)

            avg_time = data['total_time'] / data['count']

            self.debug(
                f"Performance: {operation}",
                duration=f"{duration:.3f}s",
                avg=f"{avg_time:.3f}s",
                count=data['count'],
                **metrics
            )

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._timing_lock:
            summary = {}
            for operation, data in self._performance_data.items():
                summary[operation] = {
                    'count': data['count'],
                    'avg_time': data['total_time'] / data['count'] if data['count'] > 0 else 0,
                    'min_time': data['min_time'] if data['min_time'] != float('inf') else 0,
                    'max_time': data['max_time']
                }
            return summary

    def log_camera_event(self, event: str, **details):
        """Specialized camera event logging"""
        self.info(f"Camera: {event}", **details)

    def log_error_with_traceback(self, message: str, exception: Exception):
        """Log error with full traceback"""
        import traceback
        self.error(f"{message}: {str(exception)}")
        self.debug(f"Traceback: {traceback.format_exc()}")
