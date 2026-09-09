"""Create text embeddings through OpenRouter's embeddings endpoint."""

from openai import OpenAI

from src.config import settings


client = OpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of texts into embedding vectors."""

    if not texts:
        return []
    if any(not text.strip() for text in texts):
        raise ValueError("Texts passed for embedding must not be empty")

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )

    ordered_items = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered_items]


def embed_query(query: str) -> list[float]:
    """Convert one user question into an embedding vector."""

    if not query.strip():
        raise ValueError("Query passed for embedding must not be empty")

    return embed_texts([query])[0]
