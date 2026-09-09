"""FAISS-backed vector storage for document chunks."""

from dataclasses import asdict
import json
from pathlib import Path

import faiss
import numpy as np

from src.chunker import TextChunk


class VectorStore:
    """Store and search normalized embedding vectors using cosine similarity."""

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[TextChunk] = []

    @property
    def size(self) -> int:
        """Return the number of vectors currently stored."""

        return len(self._chunks)

    def add(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        """Add chunks and their embeddings to the index."""

        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must match")
        if not chunks:
            return

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("Embeddings must be a non-empty two-dimensional array")

        faiss.normalize_L2(vectors)
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        elif self._index.d != vectors.shape[1]:
            raise ValueError("Embedding dimensions do not match the existing index")

        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[TextChunk, float]]:
        """Return the most similar chunks and their cosine scores."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if self._index is None or not self._chunks:
            return []

        query = np.asarray([query_embedding], dtype="float32")
        if query.shape[1] != self._index.d:
            raise ValueError("Query embedding dimension does not match the index")
        faiss.normalize_L2(query)

        scores, indices = self._index.search(query, min(top_k, self.size))
        return [
            (self._chunks[index], float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and chunk metadata to a directory."""

        if self._index is None:
            raise ValueError("Cannot save an empty vector store")

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        metadata = [asdict(chunk) for chunk in self._chunks]
        (path / "chunks.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a previously saved vector store."""

        path = Path(directory)
        index_path = path / "index.faiss"
        metadata_path = path / "chunks.json"
        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Vector store files not found in: {path}")

        store = cls()
        store._index = faiss.read_index(str(index_path))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        store._chunks = [TextChunk(**item) for item in metadata]
        if store._index.ntotal != len(store._chunks):
            raise ValueError("Vector index and chunk metadata contain different sizes")
        return store
