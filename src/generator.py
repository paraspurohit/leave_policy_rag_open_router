"""Generate grounded answers from retrieved document chunks."""

from openai import OpenAI

from src.chunker import TextChunk
from src.config import settings


client = OpenAI(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
)


def _format_context(results: list[tuple[TextChunk, float]]) -> str:
    """Format retrieved chunks with page references for the prompt."""

    return "\n\n".join(
        f"[Pages {chunk.page_start}-{chunk.page_end}]\n{chunk.text}"
        for chunk, _score in results
    )


def generate_answer(
    question: str,
    retrieved_chunks: list[tuple[TextChunk, float]],
    model_id: str | None = None,
) -> str:
    """Generate an answer using only the supplied retrieved context."""

    if not question.strip():
        raise ValueError("Question must not be empty")
    if not retrieved_chunks:
        return "I could not find relevant information in the policy document."

    context = _format_context(retrieved_chunks)
    response = client.chat.completions.create(
        model=model_id or settings.generation_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer questions about the supplied policy document. "
                    "Use only the provided context. If the answer is not in the "
                    "context, say that the policy does not specify it. Include "
                    "the relevant page number(s) in your answer."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )

    answer = response.choices[0].message.content
    if not answer:
        raise RuntimeError("The generation model returned an empty answer")
    return answer.strip()
