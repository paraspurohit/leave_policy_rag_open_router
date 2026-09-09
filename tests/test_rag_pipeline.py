from src.chunker import TextChunk
from src.rag_pipeline import RAGPipeline


class FakeStore:
    def add(self, chunks, embeddings):
        self.chunks = chunks

    def search(self, embedding, top_k):
        return [(TextChunk(0, "18 days leave", 1, 1), 0.95)]


def test_pipeline_answer_uses_retrieved_context(monkeypatch) -> None:
    monkeypatch.setattr("src.rag_pipeline.embed_texts", lambda texts: [[1.0]])
    monkeypatch.setattr("src.retriever.embed_query", lambda question: [1.0])
    monkeypatch.setattr(
        "src.rag_pipeline.extract_pages",
        lambda path: [],
    )
    monkeypatch.setattr("src.rag_pipeline.chunk_pages", lambda pages, **kwargs: [])
    monkeypatch.setattr(
        "src.rag_pipeline.generate_answer",
        lambda question, chunks, model_id=None: "18 days [Page 1]",
    )

    pipeline = RAGPipeline(FakeStore())
    assert pipeline.answer("How many days?") == "18 days [Page 1]"
