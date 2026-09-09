"""End-to-end retrieval-augmented generation pipeline."""

from pathlib import Path

from src.chunker import chunk_pages
from src.embeddings import embed_texts
from src.generator import generate_answer
from src.pdf_loader import extract_pages
from src.retriever import Retriever
from src.vector_store import VectorStore


class RAGPipeline:
    """Build a document index and answer questions against it."""

    def __init__(self, vector_store: VectorStore | None = None) -> None:
        self.vector_store = vector_store or VectorStore()
        self.retriever = Retriever(self.vector_store)

    def index_pdf(
        self,
        pdf_path: str | Path,
        max_characters: int = 1_200,
        overlap_characters: int = 200,
    ) -> int:
        """Extract, chunk, embed, and index a PDF. Return chunk count."""

        pages = extract_pages(pdf_path)
        chunks = chunk_pages(
            pages,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        )
        embeddings = embed_texts([chunk.text for chunk in chunks])
        self.vector_store.add(chunks, embeddings)
        return len(chunks)

    def answer(
        self,
        question: str,
        top_k: int = 5,
        model_id: str | None = None,
    ) -> str:
        """Retrieve relevant chunks and generate an answer."""

        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        return generate_answer(question, retrieved_chunks, model_id=model_id)

    def save_index(self, directory: str | Path) -> None:
        """Save the current vector index and chunk metadata."""

        self.vector_store.save(directory)

    @classmethod
    def from_saved_index(cls, directory: str | Path) -> "RAGPipeline":
        """Create a pipeline from a previously saved vector index."""

        return cls(vector_store=VectorStore.load(directory))
