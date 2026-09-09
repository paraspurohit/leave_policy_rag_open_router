"""Evaluate a RAG pipeline against the policy question set."""

import json
from pathlib import Path
import sys
from typing import Any

# Make project imports work when this file is run directly:
# `python evaluation/evaluate_models.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_pipeline import RAGPipeline
from src.model_selector import GENERATION_CANDIDATES


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation questions from a JSON file."""

    questions_path = Path(path)
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation questions must be a JSON list")
    return data


def evaluate_pipeline(
    pipeline: RAGPipeline,
    questions_path: str | Path,
    model_id: str | None = None,
) -> list[dict[str, Any]]:
    """Run every evaluation question and record basic fact matching."""

    results: list[dict[str, Any]] = []
    for item in load_questions(questions_path):
        answer = pipeline.answer(item["question"], model_id=model_id)
        answer_lower = answer.lower()
        expected_facts = item.get("expected_facts", [])
        matched_facts = [
            fact for fact in expected_facts if fact.lower() in answer_lower
        ]
        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": answer,
                "matched_facts": matched_facts,
                "fact_score": len(matched_facts) / len(expected_facts)
                if expected_facts
                else 0.0,
            }
        )
    return results


def save_results(results: list[dict[str, Any]], output_path: str | Path) -> None:
    """Write evaluation results as formatted JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_candidates(
    pipeline: RAGPipeline,
    questions_path: str | Path,
) -> list[dict[str, Any]]:
    """Evaluate every configured generation candidate."""

    comparisons: list[dict[str, Any]] = []
    for candidate in GENERATION_CANDIDATES:
        try:
            results = evaluate_pipeline(pipeline, questions_path, candidate.model_id)
            total_score = sum(item["fact_score"] for item in results)
            average_score = total_score / len(results) if results else 0.0
            comparisons.append(
                {
                    "model_id": candidate.model_id,
                    "role": candidate.role,
                    "average_fact_score": average_score,
                    "results": results,
                    "status": "success",
                }
            )
        except Exception as error:
            comparisons.append(
                {
                    "model_id": candidate.model_id,
                    "role": candidate.role,
                    "average_fact_score": 0.0,
                    "results": [],
                    "status": "error",
                    "error": str(error),
                }
            )
    return comparisons


def main() -> None:
    """Index the policy PDF, evaluate candidates, and save a report."""

    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "Leave Policy MI V1.5"
    questions_path = project_root / "evaluation" / "questions.json"
    output_path = project_root / "evaluation" / "results.json"

    pipeline = RAGPipeline()
    chunk_count = pipeline.index_pdf(pdf_path)
    comparisons = evaluate_candidates(pipeline, questions_path)
    save_results(comparisons, output_path)

    print(f"Indexed {chunk_count} chunks.")
    for comparison in comparisons:
        if comparison["status"] == "success":
            print(
                f"{comparison['model_id']}: "
                f"average fact score={comparison['average_fact_score']:.2f}"
            )
        else:
            print(f"{comparison['model_id']}: ERROR - {comparison['error']}")
    print(f"Saved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
