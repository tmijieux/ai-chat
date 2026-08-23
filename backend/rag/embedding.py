"""CPU-based text embedding for RAG spaces, via fastembed (ONNX Runtime). Runs entirely on CPU so
it never competes with the chat model or image generation for VRAM, and never needs the
stop-the-chat-model handoff generate_image pays for.

EmbeddingProvider is an abstract interface so a future GPU-based provider (for fast bulk-ingestion
of a large repo, accepting the VRAM-contention cost the CPU provider was chosen to avoid) can be
swapped in later without touching the ingestion/storage code above it. Only the CPU provider is
implemented for now.

get_embedding_provider() is keyed by model name, not a single process-wide singleton: a space
records the model it was actually embedded with (RagSpace.embedding_model), and ingestion/search
must embed using THAT model, not necessarily whatever DEFAULT_EMBEDDING_MODEL currently is — two
spaces can legitimately be on different models at once (one predating a model change, not yet
recomputed via rag_recompute.py) and each keeps working correctly. See ADR-0018.
"""
from abc import ABC, abstractmethod

DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v2-base-code"


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


_providers: dict[str, EmbeddingProvider] = {}


def get_embedding_provider(model_name: str | None = None) -> EmbeddingProvider:
    """Return the provider for `model_name` (DEFAULT_EMBEDDING_MODEL if not given), creating and
    caching it lazily on first use. Multiple models can be loaded at once — each is small (tens to
    a few hundred MB) and CPU-only, so keeping more than one resident isn't a real cost."""
    key = model_name or DEFAULT_EMBEDDING_MODEL
    if key not in _providers:
        _providers[key] = FastEmbedProvider(key)
    return _providers[key]
