from configrations import config

class IO_Manager():
    _handlers = {}
    _default_handler_key = None

    def __init__(self, ui_type=None):
        self._ui_type = ui_type

    def get_ui_type(self):
        return self._ui_type

    def set_ui_type(self, ui_type):
        self._ui_type = ui_type

    @classmethod
    def register_handler(cls, ui_key, handler_class, is_default=False):
        if not issubclass(handler_class, cls):
            raise TypeError(f"Handler class {handler_class.__name__} must be a subclass of {cls.__name__}")
        cls._handlers[ui_key] = handler_class
        if is_default:
            if cls._default_handler_key is not None:
                print(f"Warning: Overwriting default handler. Old: {cls._default_handler_key}, New: {ui_key}")
            cls._default_handler_key = ui_key


    @classmethod
    def get_instance(cls, ui_key=None, *args, **kwargs):
        actual_key = ui_key
        if actual_key is None:
            actual_key = cls._default_handler_key

        if actual_key is None:
            raise ValueError("No ui_key provided and no default handler is set.")

        handler_class = cls._handlers.get(actual_key)

        if handler_class:
            if 'ui_type' not in kwargs and issubclass(handler_class, IO_Manager):
                 kwargs['ui_type'] = actual_key
            return handler_class(*args, **kwargs)
        else:
            raise ValueError(f"No handler registered for ui_key: '{actual_key}'. Available handlers: {list(cls._handlers.keys())}")