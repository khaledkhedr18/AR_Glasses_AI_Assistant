import os
from pathlib import Path
from typing import Dict, Any
import json
from core.utils.logging import Logger
from core.utils.config_validator import ConfigValidator

class Settings:
    """Application settings manager"""

    _instance = None

    def __new__(cls):
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
        self._load_default_config()
        self._initialized = True

    def _load_default_config(self):
        """Load default configuration"""
        self._config = {
            'models': {
                'speech': {
                    'model_path': os.path.join('models', 'speech'),
                    'language': 'en'
                },
                'translation': {
                    'model_path': os.path.join('models', 'translation'),
                    'source_language': 'en',
                    'target_language': 'fr'
                }
            },
            'network': {
                'host': 'localhost',
                'port': 5000
            },
            'camera': {
                'width': 640,
                'height': 480,
                'fps': 30
            },
            'audio': {
                'sample_rate': 16000,
                'channels': 1
            }
        }

        if not self.validator.validate_config(self._config):
            self.logger.error("Invalid default configuration")
            raise ValueError("Invalid default configuration")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        try:
            return self._config[section][key]
        except KeyError:
            self.logger.warning(f"Config key not found: {section}.{key}")
            return default

    def set(self, section: str, key: str, value: Any) -> bool:
        """Set configuration value"""
        try:
            if section not in self._config:
                self._config[section] = {}

            self._config[section][key] = value

            # Validate the updated configuration
            if not self.validator.validate_config(self._config):
                self.logger.error(f"Invalid configuration after setting {section}.{key}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error setting config: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load configuration from file"""
        try:
            config_path = Path(filepath)
            if not config_path.exists():
                self.logger.error(f"Config file not found: {filepath}")
                return False

            with open(config_path, 'r') as f:
                loaded_config = json.load(f)

            # Merge with default config
            self._config.update(loaded_config)

            # Validate the merged configuration
            if not self.validator.validate_config(self._config):
                self.logger.error("Invalid configuration loaded from file")
                self._load_default_config()  # Restore defaults
                return False

            self.logger.info(f"Configuration loaded from {filepath}")
            return True

        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON in config file: {filepath}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading config from file: {e}")
            return False

    def save_to_file(self, filepath: str) -> bool:
        """Save configuration to file"""
        try:
            config_path = Path(filepath)
            config_path.parent.mkdir(parents=True, exist_ok=True)

            with open(config_path, 'w') as f:
                json.dump(self._config, f, indent=4)

            self.logger.info(f"Configuration saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving config to file: {e}")
            return False
