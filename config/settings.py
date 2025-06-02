import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import threading
from core.utils.logging import Logger
from core.utils.config_validator import ConfigValidator

class Settings:
    """Enhanced application settings manager with camera-specific configurations"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Settings, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = Logger()
        self.validator = ConfigValidator()
        self._config: Dict[str, Any] = {}
        self._config_lock = threading.Lock()
        self._load_default_config()
        self._load_environment_overrides()
        self._initialized = True

    def _load_default_config(self):
        """Load comprehensive default configuration"""
        self._config = {
            'camera': {
                # Basic camera settings
                'width': 1280,  # Optimized for Pi 4B
                'height': 720,
                'fps': 30,
                'format': 'RGB888',

                # Advanced camera controls
                'exposure': 15000,  # Microseconds
                'gain': 1.2,
                'awb_enable': True,
                'ae_enable': True,
                'flip_horizontal': True,
                'flip_vertical': True,

                # Performance settings
                'frame_duration_limits': [33333, 33333],  # 30fps
                'buffer_size': 3,
                'capture_timeout': 5.0,  # seconds

                # Processing settings
                'preprocessing': {
                    'denoise_strength': 10,
                    'threshold_value': 150,
                    'threshold_type': 'THRESH_BINARY_OTSU'
                },

                # Quality settings
                'jpeg_quality': 90,
                'jpeg_optimize': True,
                'capture_quality': 85,

                # Monitoring
                'performance_monitoring': True,
                'fps_calculation_interval': 30,  # frames
                'memory_cleanup_interval': 100,  # frames

                # Error handling
                'max_retry_attempts': 3,
                'retry_delay': 0.5,  # seconds
                'fallback_resolution': [640, 480]
            },

            'models': {
                'speech': {
                    'model_path': os.path.join('models', 'speech'),
                    'language': 'en',
                    'vosk_models': {
                        'en': '/home/pi/Desktop/gradproj/vosk-model-small-en-us-0.15',
                        'ar': '/home/pi/Desktop/gradproj/vosk-model-ar-mgb2-0.4',
                        'fr': '/home/pi/Desktop/gradproj/vosk-model-small-fr-0.22'
                    }
                },
                'translation': {
                    'model_path': os.path.join('models', 'translation'),
                    'source_language': 'en',
                    'target_language': 'fr',
                    'cache_size': 1000
                }
            },

            'network': {
                'host': 'localhost',
                'port': 5000,
                'server_ip': '192.168.1.65',
                'server_port': 4040,
                'timeout': 10,
                'max_retries': 3
            },

            'audio': {
                'sample_rate': 16000,
                'channels': 1,
                'chunk_size': 1024,
                'format': 'int16'
            },

            'logging': {
                'level': 'INFO',
                'file_rotation_size': 10485760,  # 10MB
                'backup_count': 5,
                'performance_logging': True
            },

            'performance': {
                'resource_pool_size': 10,
                'max_cached_frames': 5,
                'garbage_collection_threshold': 100
            }
        }

        if not self.validator.validate_config(self._config):
            self.logger.error("Invalid default configuration")
            raise ValueError("Invalid default configuration")

    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables"""
        try:
            # Camera overrides
            if os.getenv('CAMERA_WIDTH'):
                self.set('camera', 'width', int(os.getenv('CAMERA_WIDTH')))
            if os.getenv('CAMERA_HEIGHT'):
                self.set('camera', 'height', int(os.getenv('CAMERA_HEIGHT')))
            if os.getenv('CAMERA_FPS'):
                self.set('camera', 'fps', int(os.getenv('CAMERA_FPS')))

            # Network overrides
            if os.getenv('SERVER_IP'):
                self.set('network', 'server_ip', os.getenv('SERVER_IP'))
            if os.getenv('SERVER_PORT'):
                self.set('network', 'server_port', int(os.getenv('SERVER_PORT')))

            self.logger.info("Environment overrides applied")

        except Exception as e:
            self.logger.warning(f"Failed to apply environment overrides: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """Enhanced get with dot notation support"""
        with self._config_lock:
            try:
                if '.' in key_path:
                    keys = key_path.split('.')
                    value = self._config
                    for key in keys:
                        if isinstance(value, dict) and key in value:
                            value = value[key]
                        else:
                            self.logger.debug(f"Config key path not found: {key_path}")
                            return default
                    return value
                else:
                    return self._config.get(key_path, default)
            except (KeyError, TypeError) as e:
                self.logger.debug(f"Error getting config {key_path}: {e}")
                return default

    def set(self, section: str, key: str, value: Any) -> bool:
        """Enhanced set with validation"""
        with self._config_lock:
            try:
                if section not in self._config:
                    self._config[section] = {}

                old_value = self._config[section].get(key)
                self._config[section][key] = value

                # Validate the updated configuration
                if not self.validator.validate_config(self._config):
                    # Restore old value on validation failure
                    if old_value is not None:
                        self._config[section][key] = old_value
                    else:
                        del self._config[section][key]
                    self.logger.error(f"Invalid configuration after setting {section}.{key}")
                    return False

                self.logger.debug(f"Config updated: {section}.{key} = {value}")
                return True

            except Exception as e:
                self.logger.error(f"Error setting config: {e}")
                return False

    def get_camera_config(self) -> Dict[str, Any]:
        """Get complete camera configuration"""
        return self.get('camera', {})

    def get_camera_resolution(self) -> tuple:
        """Get camera resolution as tuple"""
        return (
            self.get('camera.width', 1280),
            self.get('camera.height', 720)
        )

    def get_camera_controls(self) -> Dict[str, Any]:
        """Get camera control settings"""
        return {
            'AwbEnable': self.get('camera.awb_enable', True),
            'AeEnable': self.get('camera.ae_enable', True),
            'ExposureTime': self.get('camera.exposure', 15000),
            'AnalogueGain': self.get('camera.gain', 1.2),
            'FrameDurationLimits': tuple(self.get('camera.frame_duration_limits', [33333, 33333]))
        }

    def get_camera_quality_settings(self) -> Dict[str, Any]:
        """Get camera quality settings"""
        return {
            'jpeg_quality': self.get('camera.jpeg_quality', 90),
            'jpeg_optimize': self.get('camera.jpeg_optimize', True),
            'capture_quality': self.get('camera.capture_quality', 85)
        }

    def update_from_dict(self, config_dict: Dict[str, Any]) -> bool:
        """Update configuration from dictionary"""
        with self._config_lock:
            try:
                # Deep merge
                self._deep_merge(self._config, config_dict)

                if not self.validator.validate_config(self._config):
                    self.logger.error("Invalid configuration after bulk update")
                    self._load_default_config()  # Restore defaults
                    return False

                self.logger.info("Configuration updated from dictionary")
                return True

            except Exception as e:
                self.logger.error(f"Error updating config from dict: {e}")
                return False

    def _deep_merge(self, target: Dict, source: Dict):
        """Deep merge two dictionaries"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value

    def load_from_file(self, filepath: str) -> bool:
        """Enhanced file loading with backup"""
        try:
            config_path = Path(filepath)
            if not config_path.exists():
                self.logger.error(f"Config file not found: {filepath}")
                return False

            # Backup current config
            backup_config = self._config.copy()

            with open(config_path, 'r') as f:
                loaded_config = json.load(f)

            # Merge with current config
            if self.update_from_dict(loaded_config):
                self.logger.info(f"Configuration loaded from {filepath}")
                return True
            else:
                # Restore backup on failure
                self._config = backup_config
                return False

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file {filepath}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading config from file: {e}")
            return False

    def save_to_file(self, filepath: str) -> bool:
        """Enhanced file saving with atomic write"""
        try:
            config_path = Path(filepath)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            # Atomic write using temporary file
            temp_path = config_path.with_suffix('.tmp')

            with open(temp_path, 'w') as f:
                json.dump(self._config, f, indent=4, sort_keys=True)

            # Move temp file to final location
            temp_path.rename(config_path)

            self.logger.info(f"Configuration saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving config to file: {e}")
            return False

    def get_all_config(self) -> Dict[str, Any]:
        """Get complete configuration (thread-safe copy)"""
        with self._config_lock:
            return self._config.copy()

    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        with self._config_lock:
            self.logger.info("Resetting configuration to defaults")
            self._load_default_config()
