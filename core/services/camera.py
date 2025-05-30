from typing import Optional, Tuple, Dict, Any, Callable
import cv2
import numpy as np
from picamera2 import Picamera2
import libcamera
from config.settings import Settings
from core.utils.logging import Logger
from core.utils.progress import ProgressMonitor
from core.utils.resource_pool import FramePool
import threading
import time
import gc
import psutil
from contextlib import contextmanager

class CameraService:
    """Ultimate enhanced camera service with full utility integration"""

    def __init__(self):
        # Initialize utilities
        self.logger = Logger()
        self.settings = Settings()
        self.progress_monitor = ProgressMonitor()

        # Frame pool for efficient memory management
        pool_size = self.settings.get('performance.max_cached_frames', 5)
        self.frame_pool = FramePool(max_size=pool_size)

        # Camera hardware
        self.picam2 = None
        self.camera_lock = threading.Lock()
        self._is_initialized = False

        # Enhanced performance monitoring
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.capture_width = 0
        self.capture_height = 0

        # Error handling and retry logic
        self.max_retries = self.settings.get('camera.max_retry_attempts', 3)
        self.retry_delay = self.settings.get('camera.retry_delay', 0.5)
        self.consecutive_errors = 0

        # Cached frame for multi-operation scenarios
        self._cached_frame = None
        self._cache_lock = threading.Lock()
        self._cache_timestamp = 0

        # Performance thresholds
        self.fps_calculation_interval = self.settings.get('camera.fps_calculation_interval', 30)
        self.memory_cleanup_interval = self.settings.get('camera.memory_cleanup_interval', 100)

        # Initialize camera
        self.initialize()

    def initialize(self) -> bool:
        """Enhanced initialization with comprehensive error handling and settings integration"""
        if self._is_initialized:
            self.logger.info("Camera already initialized")
            return True

        self.logger.info("Initializing camera service...",
                        settings_loaded=True,
                        utilities_ready=True)

        self.progress_monitor.update(10, "Initializing camera hardware...")

        for attempt in range(self.max_retries):
            try:
                if self.picam2 is None:
                    self.picam2 = Picamera2()
                    self.progress_monitor.update(30, "Camera object created")

                # Load camera configuration from settings
                camera_config = self._build_camera_config()
                self.progress_monitor.update(50, "Applying camera configuration...")

                self.picam2.configure(camera_config)

                # Apply camera controls from settings
                controls = self.settings.get_camera_controls()
                self.picam2.set_controls(controls)
                self.progress_monitor.update(70, "Camera controls applied")

                # Start camera
                self.picam2.start()
                self.progress_monitor.update(90, "Camera started")

                # Store configuration details
                self.capture_width, self.capture_height = self.settings.get_camera_resolution()

                self._is_initialized = True
                self.consecutive_errors = 0

                self.logger.log_camera_event(
                    "initialized",
                    resolution=f"{self.capture_width}x{self.capture_height}",
                    attempt=attempt + 1,
                    controls=controls
                )

                self.progress_monitor.update(100, "Camera initialization complete")
                return True

            except Exception as e:
                self.consecutive_errors += 1
                self.logger.error(f"Camera initialization attempt {attempt + 1} failed",
                                error=str(e),
                                consecutive_errors=self.consecutive_errors)

                if attempt < self.max_retries - 1:
                    self.progress_monitor.update(20 + attempt * 20, f"Retrying... ({attempt + 1}/{self.max_retries})")
                    time.sleep(self.retry_delay)
                    self._cleanup_failed_init()

        # All attempts failed
        self.progress_monitor.update(0, "Camera initialization failed")
        self.logger.critical("Camera initialization failed after all attempts")
        return False

    def _build_camera_config(self) -> Dict[str, Any]:
        """Build camera configuration from settings"""
        width, height = self.settings.get_camera_resolution()

        config = self.picam2.create_preview_configuration(
            main={
                "size": (width, height),
                "format": self.settings.get('camera.format', 'RGB888')
            },
            controls=self.settings.get_camera_controls(),
            transform=libcamera.Transform(
                hflip=self.settings.get('camera.flip_horizontal', True),
                vflip=self.settings.get('camera.flip_vertical', True)
            )
        )

        return config

    def _cleanup_failed_init(self):
        """Clean up after failed initialization"""
        try:
            if self.picam2:
                self.picam2.close()
                self.picam2 = None
        except:
            pass

    @contextmanager
    def _performance_timing(self, operation_name: str):
        """Context manager for performance timing"""
        self.progress_monitor.start_operation(operation_name)
        start_time = time.time()
        try:
            yield
        finally:
            duration = self.progress_monitor.end_operation(operation_name)
            if duration:
                self.logger.log_performance(operation_name, duration)

    def capture_frame(self) -> Optional[np.ndarray]:
        """Enhanced frame capture with performance monitoring and error recovery"""
        if not self._is_initialized or not self.picam2:
            self.logger.error("Camera not initialized for frame capture")
            return None

        with self._performance_timing("frame_capture"):
            try:
                with self.camera_lock:
                    frame = self.picam2.capture_array("main")

                    if frame is None or frame.size == 0:
                        self.dropped_frames += 1
                        self.logger.warning("Dropped frame - empty data",
                                          dropped_count=self.dropped_frames)
                        return None

                    # Update performance metrics
                    self._update_performance_metrics()

                    # Reset error counter on successful capture
                    self.consecutive_errors = 0

                    return frame

            except Exception as e:
                self.consecutive_errors += 1
                self.dropped_frames += 1

                self.logger.error("Frame capture failed",
                                error=str(e),
                                consecutive_errors=self.consecutive_errors,
                                dropped_frames=self.dropped_frames)

                # Attempt recovery if too many consecutive errors
                if self.consecutive_errors >= 5:
                    self.logger.warning("Attempting camera recovery...")
                    self._attempt_recovery()

                return None

    def _update_performance_metrics(self):
        """Update performance metrics and monitoring"""
        self.frame_count += 1

        # Calculate FPS
        if self.frame_count % self.fps_calculation_interval == 0:
            current_time = time.time()
            elapsed = current_time - self.last_fps_time

            if elapsed > 0:
                self.fps = self.fps_calculation_interval / elapsed
                self.last_fps_time = current_time

                # Update progress monitor
                self.progress_monitor.update_camera_metrics(
                    fps=self.fps,
                    frame_count=self.frame_count,
                    dropped_frames=self.dropped_frames
                )

        # Memory cleanup
        if self.frame_count % self.memory_cleanup_interval == 0:
            gc.collect()

            # Log memory usage
            memory_percent = psutil.virtual_memory().percent
            self.logger.debug("Memory cleanup performed",
                            frame_count=self.frame_count,
                            memory_usage=f"{memory_percent:.1f}%")

    def _attempt_recovery(self):
        """Attempt to recover from camera errors"""
        try:
            self.logger.info("Attempting camera recovery...")

            with self.camera_lock:
                if self.picam2:
                    self.picam2.stop()
                    time.sleep(1)
                    self.picam2.start()

            self.consecutive_errors = 0
            self.logger.info("Camera recovery successful")

        except Exception as e:
            self.logger.error(f"Camera recovery failed: {e}")
            self._is_initialized = False

    def capture_frame_rgb(self) -> Optional[np.ndarray]:
        """Enhanced RGB conversion with error handling"""
        frame = self.capture_frame()
        if frame is None:
            return None

        with self._performance_timing("rgb_conversion"):
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return rgb_frame
            except Exception as e:
                self.logger.error("RGB conversion failed", error=str(e))
                return None

    def capture_and_resize(self, width: int, height: int, rgb: bool = False) -> Optional[np.ndarray]:
        """Enhanced capture and resize with settings-based optimization"""
        frame = self.capture_frame_rgb() if rgb else self.capture_frame()
        if frame is None:
            return None

        return self.resize_frame(frame, width, height)

    def resize_frame(self, frame: np.ndarray, width: int, height: int) -> Optional[np.ndarray]:
        """Enhanced resize with settings-based interpolation"""
        if frame is None or frame.size == 0:
            self.logger.error("Invalid frame for resizing")
            return None

        with self._performance_timing("frame_resize"):
            try:
                # Choose interpolation based on resize ratio
                original_size = frame.shape[0] * frame.shape[1]
                target_size = width * height

                if target_size < original_size * 0.5:
                    interpolation = cv2.INTER_LANCZOS4
                else:
                    interpolation = cv2.INTER_LINEAR

                resized = cv2.resize(frame, (width, height), interpolation=interpolation)
                return resized

            except Exception as e:
                self.logger.error("Frame resize failed",
                                error=str(e),
                                target_size=f"{width}x{height}")
                return None

    def capture_image_to_file(self, filename: str = "captured_image.jpg") -> bool:
        """Enhanced image capture with settings-based quality"""
        if not self._is_initialized:
            self.logger.error("Camera not initialized for image capture")
            return False

        with self._performance_timing("image_capture"):
            try:
                with self.camera_lock:
                    buffer = self.picam2.capture_array("main")

                    if buffer is None or buffer.size == 0:
                        self.logger.error("Failed to capture image - empty buffer")
                        return False

                    # Get quality settings from configuration
                    quality_settings = self.settings.get_camera_quality_settings()

                    cv2_params = [
                        cv2.IMWRITE_JPEG_QUALITY, quality_settings['jpeg_quality'],
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1 if quality_settings['jpeg_optimize'] else 0
                    ]

                    success = cv2.imwrite(filename, buffer, cv2_params)

                    if success:
                        file_size = len(cv2.imencode('.jpg', buffer, cv2_params)[1])
                        self.logger.log_camera_event(
                            "image_saved",
                            filename=filename,
                            size_bytes=file_size,
                            quality=quality_settings['jpeg_quality']
                        )
                        return True
                    else:
                        self.logger.error("Failed to save image", filename=filename)
                        return False

            except Exception as e:
                self.logger.log_error_with_traceback("Image capture failed", e)
                return False

    def get_frame_for_processing(self) -> Optional[np.ndarray]:
        """Enhanced preprocessing with settings-based parameters"""
        frame = self.capture_frame()
        if frame is None:
            return None

        with self._performance_timing("frame_preprocessing"):
            try:
                # Get preprocessing settings
                preprocessing = self.settings.get('camera.preprocessing', {})

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Apply denoising
                denoise_strength = preprocessing.get('denoise_strength', 10)
                denoised = cv2.fastNlMeansDenoising(gray, h=denoise_strength)

                # Apply thresholding
                threshold_value = preprocessing.get('threshold_value', 150)
                threshold_type = getattr(cv2, preprocessing.get('threshold_type', 'THRESH_BINARY_OTSU'))

                _, thresh = cv2.threshold(denoised, threshold_value, 255, threshold_type)

                return thresh

            except Exception as e:
                self.logger.error("Frame preprocessing failed", error=str(e))
                return frame

    def cache_current_frame(self, max_age: float = 1.0) -> bool:
        """Enhanced frame caching with age management"""
        frame = self.capture_frame()
        if frame is None:
            return False

        try:
            with self._cache_lock:
                self._cached_frame = frame.copy()
                self._cache_timestamp = time.time()

            self.logger.debug("Frame cached successfully")
            return True

        except Exception as e:
            self.logger.error("Frame caching failed", error=str(e))
            return False

    def get_cached_frame(self, max_age: float = 1.0) -> Optional[np.ndarray]:
        """Get cached frame with age validation"""
        with self._cache_lock:
            if self._cached_frame is None:
                return None

            age = time.time() - self._cache_timestamp
            if age > max_age:
                self.logger.debug("Cached frame too old", age=f"{age:.2f}s")
                self._cached_frame = None
                return None

            return self._cached_frame.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Enhanced performance statistics"""
        camera_stats = self.progress_monitor.get_camera_performance()

        return {
            **camera_stats,
            "resolution": f"{self.capture_width}x{self.capture_height}",
            "is_initialized": self._is_initialized,
            "consecutive_errors": self.consecutive_errors,
            "memory_usage": psutil.virtual_memory().percent,
            "frame_pool_stats": self.frame_pool.get_stats()
        }

    def get_detailed_info(self) -> Dict[str, Any]:
        """Get comprehensive camera information"""
        settings_summary = {
            'resolution': self.settings.get_camera_resolution(),
            'controls': self.settings.get_camera_controls(),
            'quality': self.settings.get_camera_quality_settings()
        }

        return {
            "camera_info": {
                "width": self.capture_width,
                "height": self.capture_height,
                "format": self.settings.get('camera.format', 'RGB888'),
                "fps_target": self.settings.get('camera.fps', 30),
                "actual_fps": self.fps
            },
            "performance": self.get_performance_stats(),
            "settings": settings_summary,
            "status": {
                "initialized": self._is_initialized,
                "error_recovery_needed": self.consecutive_errors >= 5
            }
        }

    def cleanup(self):
        """Enhanced cleanup with full resource management"""
        self.logger.info("Starting comprehensive camera cleanup...")

        try:
            # Stop progress monitoring
            if hasattr(self, 'progress_monitor'):
                self.progress_monitor.reset()

            # Clean up camera hardware
            with self.camera_lock:
                if self.picam2:
                    self.picam2.stop()
                    self.picam2.close()
                    self.picam2 = None

            # Clear cached data
            with self._cache_lock:
                self._cached_frame = None

            # Cleanup frame pool
            if hasattr(self, 'frame_pool'):
                self.frame_pool.cleanup_all()

            self._is_initialized = False

            # Log final statistics
            final_stats = {
                'total_frames': self.frame_count,
                'dropped_frames': self.dropped_frames,
                'final_fps': self.fps,
                'consecutive_errors': self.consecutive_errors
            }

            self.logger.log_camera_event("cleanup_completed", **final_stats)

        except Exception as e:
            self.logger.log_error_with_traceback("Error during camera cleanup", e)
        finally:
            self._is_initialized = False

    def __enter__(self):
        """Enhanced context manager with initialization check"""
        if not self.is_ready():
            if not self.initialize():
                raise RuntimeError("Failed to initialize camera service")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup with exception logging"""
        if exc_type:
            self.logger.error("Exception in camera context",
                            exception_type=exc_type.__name__,
                            exception_message=str(exc_val))
        self.cleanup()

    def is_ready(self) -> bool:
        """Enhanced readiness check"""
        return (self._is_initialized and
                self.picam2 is not None and
                self.consecutive_errors < 5)

    def get_progress_monitor(self) -> ProgressMonitor:
        """Get the progress monitor instance"""
        return self.progress_monitor

    def __del__(self):
        """Enhanced destructor with error suppression"""
        try:
            self.cleanup()
        except:
            # Suppress all errors during destruction
            pass
