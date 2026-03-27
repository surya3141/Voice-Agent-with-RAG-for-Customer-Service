"""Shared test fixtures and configuration."""

import chromadb
from chromadb.api.models.Collection import Collection


class DummyEmbeddingFunction(chromadb.EmbeddingFunction):
    """A simple deterministic embedding function for offline tests."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            h = hash(text)
            embedding = [((h >> (i % 64)) & 0xFF) / 255.0 for i in range(384)]
            embeddings.append(embedding)
        return embeddings


_dummy_ef = DummyEmbeddingFunction()

# Patch Collection.__init__ to inject a dummy embedding function when none is set
_orig_collection_init = Collection.__init__


def _patched_collection_init(self, *args, **kwargs):
    _orig_collection_init(self, *args, **kwargs)
    if self._embedding_function is None or type(self._embedding_function).__name__ == "ONNXMiniLM_L6_V2":
        self._embedding_function = _dummy_ef


Collection.__init__ = _patched_collection_init
