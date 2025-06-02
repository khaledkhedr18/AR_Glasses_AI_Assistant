import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
import tempfile
import os
from pathlib import Path

@pytest.fixture
def mock_camera():
    """Mock camera for testing without hardware"""
    mock = Mock()
    mock.capture_array.return_value = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    mock.configure = Mock()
    mock.set_controls = Mock()
    mock.start = Mock()
    mock.stop = Mock()
    mock.close = Mock()
    return mock

@pytest.fixture
def sample_frame():
    """Generate a sample camera frame for testing"""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

@pytest.fixture
def test_config():
    """Test configuration for camera service"""
    return {
        'camera': {
            'width': 640,
            'height': 480,
            'fps': 30,
            'format': 'RGB888',
            'exposure': 15000,
            'gain': 1.2,
            'preprocessing': {
                'denoise_strength': 10,
                'threshold_value': 150
            }
        },
        'performance': {
            'max_cached_frames': 3
        }
    }

@pytest.fixture
def temp_directory():
    """Temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def test_image_path():
    """Path to test image files"""
    return Path(__file__).parent / "fixtures" / "sample_images"
