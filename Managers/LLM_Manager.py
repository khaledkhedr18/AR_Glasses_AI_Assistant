from configrations import config

class LLM_Manager():
    _handlers = {}
    _default_handler_key = None

    def __init__(self, purpose=None):
        self._purpose = purpose

    def get_purpose(self):
        return self._purpose

    def set_purpose(self, purpose):
        self._purpose = purpose

    @classmethod
    def register_handler(cls, purpose_key, handler_class, is_default=False):
        if not issubclass(handler_class, cls):
            raise TypeError(f"Handler class {handler_class.__name__} must be a subclass of {cls.__name__}")
        cls._handlers[purpose_key] = handler_class
        if is_default:
            if cls._default_handler_key is not None:
                print(f"Warning: Overwriting default handler. Old: {cls._default_handler_key}, New: {purpose_key}")
            cls._default_handler_key = purpose_key


    @classmethod
    def get_instance(cls, purpose_key=None, *args, **kwargs):
        actual_key = purpose_key
        if actual_key is None:
            actual_key = cls._default_handler_key

        if actual_key is None:
            raise ValueError("No purpose_key provided and no default handler is set.")

        handler_class = cls._handlers.get(actual_key)

        if handler_class:
            if 'purpose' not in kwargs and issubclass(handler_class, LLM_Manager):
                 kwargs['purpose'] = actual_key
            return handler_class(*args, **kwargs)
        else:
            raise ValueError(f"No handler registered for purpose: '{actual_key}'. Available handlers: {list(cls._handlers.keys())}")

