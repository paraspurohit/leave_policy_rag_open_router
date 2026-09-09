"""Candidate model definitions and evaluation-based model selection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelCandidate:
    """A model that can be evaluated for one RAG role."""

    model_id: str
    role: str
    reason: str


GENERATION_CANDIDATES = [
    ModelCandidate(
        model_id="liquid/lfm-2.5-2.6b:free",
        role="generation",
        reason="Free compact model listed for RAG, data extraction, and long-context tasks.",
    ),
    ModelCandidate(
        model_id="thinkingmachines/inkling:free",
        role="generation",
        reason="Larger Inkling model with long-context and reasoning capability.",
    ),
    ModelCandidate(
        model_id="nvidia/nemotron-3-ultra:free",
        role="generation",
        reason="Strong long-context reasoning for complex document questions.",
    ),
]


EMBEDDING_CANDIDATES = [
    ModelCandidate(
        model_id="liquid/lfm-2.5-embedding-350m:free",
        role="embedding",
        reason="Free embedding model intended for semantic search and retrieval.",
    ),
]


def choose_best(
    candidates: list[ModelCandidate],
    scores: dict[str, float],
) -> ModelCandidate:
    """Choose the highest-scoring candidate after evaluation."""

    if not candidates:
        raise ValueError("At least one model candidate is required")
    missing_scores = [candidate.model_id for candidate in candidates if candidate.model_id not in scores]
    if missing_scores:
        raise ValueError(f"Missing evaluation scores for: {', '.join(missing_scores)}")

    return max(candidates, key=lambda candidate: scores[candidate.model_id])
