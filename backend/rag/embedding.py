"""CPU-based text embedding for RAG spaces, via fastembed (ONNX Runtime). Runs entirely on CPU so
it never competes with the chat model or image generation for VRAM, and never needs the
stop-the-chat-model handoff generate_image pays for.

EmbeddingProvider is an abstract interface so a future GPU-based provider (for fast bulk-ingestion
of a large repo, accepting the VRAM-contention cost the CPU provider was chosen to avoid) can be
swapped in later without touching the ingestion/storage code above it. Only the CPU provider is
implemented for now.
"""
from abc import ABC, abstractmethod

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class EmbeddingProvider(ABC):
    """A source of text embeddings for RAG chunking/search."""

    model_name: str
    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input in the same order."""
        ...


class FastEmbedProvider(EmbeddingProvider):
    """CPU embedding provider backed by fastembed (ONNX Runtime, no torch dependency)."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        """Load the given fastembed model and probe its output dimension."""
        from fastembed import TextEmbedding
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self.dimension = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts, returning one vector per input in the same order."""
        if len(texts) == 0:
            return []
        return [vector.tolist() for vector in self._model.embed(texts)]


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide embedding provider, creating it lazily on first use."""
    global _provider
    if _provider is None:
        _provider = FastEmbedProvider()
    return _provider
