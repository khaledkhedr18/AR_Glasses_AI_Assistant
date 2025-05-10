from typing import Dict, Any, List, Optional
from core.utils.logging import Logger

class ConfigValidator:
    """Utility for validating configuration settings"""

    def __init__(self):
        self.logger = Logger()
        self._required_fields: Dict[str, List[str]] = {
            'models': ['speech', 'translation'],
            'network': ['host', 'port'],
            'camera': ['width', 'height', 'fps'],
            'audio': ['sample_rate', 'channels']
        }

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary"""
        try:
            # Check required sections
            for section, fields in self._required_fields.items():
                if section not in config:
                    self.logger.error(f"Missing required section: {section}")
                    return False

                # Check required fields in section
                for field in fields:
                    if field not in config[section]:
                        self.logger.error(f"Missing required field '{field}' in section '{section}'")
                        return False

            # Validate specific field types and values
            if not self._validate_network_config(config['network']):
                return False
            if not self._validate_camera_config(config['camera']):
                return False
            if not self._validate_audio_config(config['audio']):
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating config: {e}")
            return False

    def _validate_network_config(self, config: Dict[str, Any]) -> bool:
        """Validate network configuration"""
        try:
            # Validate host
            host = config['host']
            if not isinstance(host, str) or not host:
                self.logger.error("Invalid host value")
                return False

            # Validate port
            port = config['port']
            if not isinstance(port, int) or not (1024 <= port <= 65535):
                self.logger.error("Invalid port value")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating network config: {e}")
            return False

    def _validate_camera_config(self, config: Dict[str, Any]) -> bool:
        """Validate camera configuration"""
        try:
            # Validate width
            width = config['width']
            if not isinstance(width, int) or width <= 0:
                self.logger.error("Invalid camera width")
                return False

            # Validate height
            height = config['height']
            if not isinstance(height, int) or height <= 0:
                self.logger.error("Invalid camera height")
                return False

            # Validate fps
            fps = config['fps']
            if not isinstance(fps, int) or not (1 <= fps <= 60):
                self.logger.error("Invalid camera fps")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating camera config: {e}")
            return False

    def _validate_audio_config(self, config: Dict[str, Any]) -> bool:
        """Validate audio configuration"""
        try:
            # Validate sample rate
            sample_rate = config['sample_rate']
            if not isinstance(sample_rate, int) or sample_rate not in [8000, 16000, 44100, 48000]:
                self.logger.error("Invalid sample rate")
                return False

            # Validate channels
            channels = config['channels']
            if not isinstance(channels, int) or channels not in [1, 2]:
                self.logger.error("Invalid channel count")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating audio config: {e}")
            return False
