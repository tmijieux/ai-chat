from .base import SttBackend
from .openvino_backend import OpenVinoBackend
from .whisper_cpp_backend import WhisperCppBackend

# Switch this one line to change backends
backend: SttBackend = WhisperCppBackend()
