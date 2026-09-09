from src.chunker import TextChunk
from src.retriever import Retriever


class FakeStore:
    def search(self, embedding: list[float], top_k: int):
        assert embedding == [1.0, 2.0]
        return [(TextChunk(0, "policy text", 1, 1), 0.9)][:top_k]


def test_retriever_embeds_question(monkeypatch) -> None:
    monkeypatch.setattr("src.retriever.embed_query", lambda question: [1.0, 2.0])
    results = Retriever(FakeStore()).retrieve("What is the policy?", top_k=1)

    assert len(results) == 1
    assert results[0][0].text == "policy text"
