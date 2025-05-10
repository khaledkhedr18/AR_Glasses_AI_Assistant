from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type
from core.utils.logging import Logger

class BaseModel(ABC):
    """Base class for all models in the application"""

    def __init__(self, model_path: str, **kwargs):
        self.logger = Logger()
        self.model_path = model_path
        self._model = None
        self._initialize_model(**kwargs)

    @abstractmethod
    def _initialize_model(self, **kwargs):
        """Initialize the model with given parameters"""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Make predictions using the model"""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up model resources"""
        pass

    def is_initialized(self) -> bool:
        """Check if model is properly initialized"""
        return self._model is not None

class ModelFactory:
    """Factory for creating model instances"""

    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]):
        """Register a new model type"""
        cls._models[model_type] = model_class

    @classmethod
    def get_model(cls, model_type: str, model_path: str, **kwargs) -> BaseModel:
        """Get a model instance of the specified type"""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")

        model_class = cls._models[model_type]
        return model_class(model_path, **kwargs)

# Register model types
from .speech import SpeechModel
from .translation import TranslationModel

ModelFactory.register_model('speech', SpeechModel)
ModelFactory.register_model('translation', TranslationModel)
