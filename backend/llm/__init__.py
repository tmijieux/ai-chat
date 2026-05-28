from .base import LLMBackend
from .ollama import OllamaBackend
from .llama_server import LlamaServerBackend

# Switch this one line to change backends
backend: LLMBackend = LlamaServerBackend()
