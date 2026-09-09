"""Small manual entry point for trying the RAG pipeline."""

from pathlib import Path

from src.rag_pipeline import RAGPipeline


def main() -> None:
    pdf_path = Path("Leave Policy MI V1.5")
    pipeline = RAGPipeline()
    chunk_count = pipeline.index_pdf(pdf_path)
    print(f"Indexed {chunk_count} chunks.")

    question = input("Ask a question about the leave policy: ").strip()
    if question:
        print(pipeline.answer(question))


if __name__ == "__main__":
    main()
