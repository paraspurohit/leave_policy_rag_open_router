"""Gradio web interface for the leave-policy RAG assistant."""

from pathlib import Path

import gradio as gr

from src.rag_pipeline import RAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_PATH = PROJECT_ROOT / "Leave Policy MI V1.5"

pipeline: RAGPipeline | None = None
startup_message = "The assistant is not indexed yet."


def initialize() -> str:
    """Index the policy document once when the app starts."""

    global pipeline, startup_message
    pipeline = RAGPipeline()
    chunk_count = pipeline.index_pdf(PDF_PATH)
    startup_message = f"Policy loaded successfully ({chunk_count} chunks indexed)."
    return startup_message


def answer_question(question: str) -> str:
    """Answer one question from the indexed policy."""

    if not question.strip():
        return "Please enter a question."
    if pipeline is None:
        return "The policy is not ready yet. Restart the app and try again."
    return pipeline.answer(question)


def main() -> None:
    status = initialize()
    interface = gr.Interface(
        fn=answer_question,
        inputs=gr.Textbox(
            label="Your question",
            placeholder="How many Privilege Leaves are provided?",
        ),
        outputs=gr.Markdown(label="Answer"),
        title="Leave Policy Assistant",
        description=f"{status} Ask questions using the policy document.",
        examples=[
            ["How many Privilege Leaves are provided?"],
            ["How many days in advance must I apply for Privilege Leave?"],
            ["What happens to unused Casual Leave?"],
        ],
    )
    interface.launch()


if __name__ == "__main__":
    main()
