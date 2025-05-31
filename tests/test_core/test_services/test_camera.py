import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import threading
import time

# Import your camera service
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from core.services.camera import CameraService

class TestCameraService:
    """Comprehensive tests for CameraService"""

    @pytest.fixture
    def camera_service(self, test_config):
        """Create camera service with test configuration"""
        with patch('core.services.camera.Picamera2') as mock_picam2:
            mock_instance = Mock()
            mock_picam2.return_value = mock_instance

            with patch('config.settings.Settings') as mock_settings:
                mock_settings_instance = Mock()
                mock_settings_instance.get.side_effect = lambda key, default=None: test_config.get(key.split('.')[0], {}).get(key.split('.')[-1], default)
                mock_settings_instance.get_camera_resolution.return_value = (640, 480)
                mock_settings_instance.get_camera_controls.return_value = {'AwbEnable': True}
                mock_settings_instance.get_camera_quality_settings.return_value = {'jpeg_quality': 90, 'jpeg_optimize': True}
                mock_settings.return_value = mock_settings_instance

                service = CameraService()
                service.picam2 = mock_instance
                service._is_initialized = True
                yield service

    def test_camera_initialization(self, camera_service):
        """Test camera initialization process"""
        assert camera_service.is_ready()
        assert camera_service._is_initialized
        assert camera_service.picam2 is not None

    def test_frame_capture_success(self, camera_service, sample_frame):
        """Test successful frame capture"""
        camera_service.picam2.capture_array.return_value = sample_frame

        frame = camera_service.capture_frame()

        assert frame is not None
        assert frame.shape == sample_frame.shape
        assert camera_service.consecutive_errors == 0

    def test_frame_capture_failure(self, camera_service):
        """Test frame capture failure handling"""
        camera_service.picam2.capture_array.side_effect = Exception("Camera error")

        frame = camera_service.capture_frame()

        assert frame is None
        assert camera_service.consecutive_errors > 0
        assert camera_service.dropped_frames > 0

    def test_frame_capture_empty_data(self, camera_service):
        """Test handling of empty frame data"""
        camera_service.picam2.capture_array.return_value = np.array([])

        frame = camera_service.capture_frame()

        assert frame is None
        assert camera_service.dropped_frames > 0

    def test_rgb_conversion(self, camera_service, sample_frame):
        """Test BGR to RGB conversion"""
        camera_service.picam2.capture_array.return_value = sample_frame

        rgb_frame = camera_service.capture_frame_rgb()

        assert rgb_frame is not None
        assert rgb_frame.shape == sample_frame.shape

    def test_frame_resize(self, camera_service, sample_frame):
        """Test frame resizing functionality"""
        target_width, target_height = 320, 240

        resized = camera_service.resize_frame(sample_frame, target_width, target_height)

        assert resized is not None
        assert resized.shape == (target_height, target_width, 3)

    def test_capture_and_resize(self, camera_service, sample_frame):
        """Test combined capture and resize"""
        camera_service.picam2.capture_array.return_value = sample_frame

        resized = camera_service.capture_and_resize(320, 240, rgb=True)

        assert resized is not None
        assert resized.shape == (240, 320, 3)

    def test_image_capture_to_file(self, camera_service, sample_frame, temp_directory):
        """Test image capture to file"""
        camera_service.picam2.capture_array.return_value = sample_frame
        test_file = temp_directory / "test_image.jpg"

        success = camera_service.capture_image_to_file(str(test_file))

        assert success
        # Note: cv2.imwrite is mocked, so file won't actually exist in tests

    def test_frame_caching(self, camera_service, sample_frame):
        """Test frame caching functionality"""
        camera_service.picam2.capture_array.return_value = sample_frame

        # Cache a frame
        success = camera_service.cache_current_frame()
        assert success

        # Retrieve cached frame
        cached = camera_service.get_cached_frame()
        assert cached is not None
        assert np.array_equal(cached, sample_frame)

    def test_cached_frame_expiry(self, camera_service, sample_frame):
        """Test cached frame expiry"""
        camera_service.picam2.capture_array.return_value = sample_frame

        # Cache frame with very short max age
        camera_service.cache_current_frame()

        # Wait for expiry
        time.sleep(0.1)

        # Should return None for expired cache
        cached = camera_service.get_cached_frame(max_age=0.05)
        assert cached is None

    def test_frame_preprocessing(self, camera_service, sample_frame):
        """Test frame preprocessing for OCR"""
        camera_service.picam2.capture_array.return_value = sample_frame

        processed = camera_service.get_frame_for_processing()

        assert processed is not None
        assert len(processed.shape) == 2  # Should be grayscale

    def test_performance_metrics_update(self, camera_service, sample_frame):
        """Test performance metrics tracking"""
        camera_service.picam2.capture_array.return_value = sample_frame
        initial_count = camera_service.frame_count

        # Capture multiple frames
        for _ in range(5):
            camera_service.capture_frame()

        assert camera_service.frame_count == initial_count + 5

    def test_error_recovery(self, camera_service):
        """Test automatic error recovery"""
        # Simulate consecutive errors
        camera_service.consecutive_errors = 5

        # Mock successful recovery
        camera_service.picam2.stop = Mock()
        camera_service.picam2.start = Mock()

        camera_service._attempt_recovery()

        assert camera_service.consecutive_errors == 0

    def test_thread_safety(self, camera_service, sample_frame):
        """Test thread safety of camera operations"""
        camera_service.picam2.capture_array.return_value = sample_frame
        results = []

        def capture_frames():
            for _ in range(10):
                frame = camera_service.capture_frame()
                results.append(frame is not None)

        # Run multiple threads
        threads = [threading.Thread(target=capture_frames) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All captures should succeed
        assert all(results)

    def test_context_manager(self, test_config):
        """Test camera service as context manager"""
        with patch('core.services.camera.Picamera2'):
            with patch('config.settings.Settings'):
                with CameraService() as camera:
                    assert camera.is_ready()

    def test_performance_stats(self, camera_service):
        """Test performance statistics retrieval"""
        stats = camera_service.get_performance_stats()

        assert 'fps' in stats
        assert 'frame_count' in stats
        assert 'resolution' in stats
        assert 'is_initialized' in stats

    def test_detailed_info(self, camera_service):
        """Test detailed camera information"""
        info = camera_service.get_detailed_info()

        assert 'camera_info' in info
        assert 'performance' in info
        assert 'settings' in info
        assert 'status' in info

    def test_cleanup(self, camera_service):
        """Test proper cleanup"""
        camera_service.cleanup()

        assert not camera_service._is_initialized
        assert camera_service.picam2 is None or camera_service.picam2.stop.called
