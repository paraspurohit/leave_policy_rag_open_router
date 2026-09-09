"""Retrieve document chunks relevant to a user question."""

from src.embeddings import embed_query
from src.chunker import TextChunk
from src.vector_store import VectorStore


class Retriever:
    """Use the embedding model and vector store for semantic search."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[tuple[TextChunk, float]]:
        """Return the most relevant chunks and similarity scores."""

        if not question.strip():
            raise ValueError("Question must not be empty")

        question_embedding = embed_query(question)
        return self.vector_store.search(question_embedding, top_k=top_k)
