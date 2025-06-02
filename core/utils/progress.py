from typing import List, Callable, Optional, Dict, Any
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from core.utils.logging import Logger
import threading
import time

class ProgressMonitor(QObject):
    """Enhanced progress monitor with camera-specific tracking"""

    progress_updated = pyqtSignal(int, str)
    fps_updated = pyqtSignal(float)
    performance_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._progress = 0
        self._status = ""
        self._observers: List[Callable[[int, str], None]] = []
        self._performance_observers: List[Callable[[Dict[str, Any]], None]] = []

        self.logger = Logger()
        self._lock = threading.Lock()

        # Camera-specific monitoring
        self._camera_metrics = {
            'fps': 0.0,
            'frame_count': 0,
            'dropped_frames': 0,
            'capture_time': 0.0,
            'processing_time': 0.0,
            'memory_usage': 0.0
        }

        # Performance tracking
        self._operation_times: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

        # Auto-update timer
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._emit_performance_update)
        self._update_timer.start(1000)  # Update every second

    @property
    def progress(self) -> int:
        """Get current progress percentage"""
        with self._lock:
            return self._progress

    @property
    def status(self) -> str:
        """Get current status message"""
        with self._lock:
            return self._status

    def update(self, progress: int, status: str):
        """Update progress and status"""
        with self._lock:
            self._progress = max(0, min(100, progress))
            self._status = status

        self.progress_updated.emit(self._progress, self._status)
        self.notify_observers()
        self.logger.debug(f"Progress: {self._progress}% - {self._status}")

    def update_camera_metrics(self, **metrics):
        """Update camera-specific metrics"""
        with self._lock:
            for key, value in metrics.items():
                if key in self._camera_metrics:
                    self._camera_metrics[key] = value

        # Emit FPS updates separately for real-time monitoring
        if 'fps' in metrics:
            self.fps_updated.emit(metrics['fps'])

    def start_operation(self, operation_name: str):
        """Start timing an operation"""
        with self._lock:
            self._start_times[operation_name] = time.time()

    def end_operation(self, operation_name: str) -> Optional[float]:
        """End timing an operation and return duration"""
        with self._lock:
            if operation_name in self._start_times:
                duration = time.time() - self._start_times[operation_name]
                del self._start_times[operation_name]

                # Store operation time
                if operation_name not in self._operation_times:
                    self._operation_times[operation_name] = []

                self._operation_times[operation_name].append(duration)

                # Keep only last 100 measurements
                if len(self._operation_times[operation_name]) > 100:
                    self._operation_times[operation_name] = self._operation_times[operation_name][-100:]

                self.logger.log_performance(operation_name, duration)
                return duration

        return None

    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        with self._lock:
            if operation_name not in self._operation_times or not self._operation_times[operation_name]:
                return {}

            times = self._operation_times[operation_name]
            return {
                'count': len(times),
                'avg': sum(times) / len(times),
                'min': min(times),
                'max': max(times),
                'recent': times[-1] if times else 0.0
            }

    def get_camera_performance(self) -> Dict[str, Any]:
        """Get camera performance metrics"""
        with self._lock:
            metrics = self._camera_metrics.copy()

            # Add calculated metrics
            if metrics['frame_count'] > 0:
                metrics['drop_rate'] = (metrics['dropped_frames'] / metrics['frame_count']) * 100
            else:
                metrics['drop_rate'] = 0.0

            return metrics

    def add_observer(self, observer: Callable[[int, str], None]):
        """Add progress observer"""
        if observer not in self._observers:
            self._observers.append(observer)

    def add_performance_observer(self, observer: Callable[[Dict[str, Any]], None]):
        """Add performance observer"""
        if observer not in self._performance_observers:
            self._performance_observers.append(observer)

    def remove_observer(self, observer: Callable[[int, str], None]):
        """Remove progress observer"""
        if observer in self._observers:
            self._observers.remove(observer)

    def remove_performance_observer(self, observer: Callable[[Dict[str, Any]], None]):
        """Remove performance observer"""
        if observer in self._performance_observers:
            self._performance_observers.remove(observer)

    def notify_observers(self):
        """Notify all progress observers"""
        for observer in self._observers:
            try:
                observer(self._progress, self._status)
            except Exception as e:
                self.logger.error(f"Error notifying progress observer: {e}")

    def _emit_performance_update(self):
        """Emit performance update signal"""
        try:
            performance_data = {
                'camera': self.get_camera_performance(),
                'operations': {op: self.get_operation_stats(op)
                             for op in self._operation_times.keys()}
            }

            self.performance_updated.emit(performance_data)

            # Notify performance observers
            for observer in self._performance_observers:
                try:
                    observer(performance_data)
                except Exception as e:
                    self.logger.error(f"Error notifying performance observer: {e}")

        except Exception as e:
            self.logger.error(f"Error emitting performance update: {e}")

    def reset(self):
        """Reset progress and metrics"""
        with self._lock:
            self._progress = 0
            self._status = ""
            self._camera_metrics = {
                'fps': 0.0,
                'frame_count': 0,
                'dropped_frames': 0,
                'capture_time': 0.0,
                'processing_time': 0.0,
                'memory_usage': 0.0
            }
            self._operation_times.clear()
            self._start_times.clear()

        self.update(0, "")

    def get_summary_report(self) -> str:
        """Generate a summary report"""
        with self._lock:
            report_lines = [
                f"Progress: {self._progress}%",
                f"Status: {self._status}",
                f"Camera FPS: {self._camera_metrics['fps']:.1f}",
                f"Frame Count: {self._camera_metrics['frame_count']}",
                f"Drop Rate: {self.get_camera_performance()['drop_rate']:.1f}%",
                "",
                "Operation Performance:"
            ]

            for op_name, times in self._operation_times.items():
                if times:
                    stats = self.get_operation_stats(op_name)
                    report_lines.append(
                        f"  {op_name}: avg={stats['avg']:.3f}s, "
                        f"min={stats['min']:.3f}s, max={stats['max']:.3f}s"
                    )

            return "\n".join(report_lines)
