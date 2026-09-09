# Free OpenRouter RAG Platform

This project answers questions from the supplied leave-policy PDF using retrieval-augmented generation (RAG).

## Architecture

```text
PDF → extraction → chunks → embeddings → FAISS search → free LLM answer
```

- `PyMuPDF` extracts text and page numbers.
- Python code creates page-aware text chunks.
- Chunks default to 1,200 characters so they remain within the embedding model's 512-token input limit.
- `LFM2.5-Embedding-350M` creates vectors through OpenRouter.
- FAISS performs similarity search.
- `LFM2.5-2.6B` generates an answer using retrieved policy context.

The selected OpenRouter model IDs are configured in `.env`. Free-model availability and rate limits can change.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your OpenRouter API key.

3. Place PDFs in `data/raw/` or pass a PDF path directly to the pipeline.

## Example

```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.index_pdf("Leave Policy MI V1.5")
print(pipeline.answer("How many Privilege Leaves are provided?"))
```

## Evaluation

Evaluation questions are stored in `evaluation/questions.json`. Results can be generated with `evaluate_pipeline()` from `evaluation/evaluate_models.py`.

## Tests

```bash
pytest
```

## Gradio interface

Install the added Gradio dependency and start the web interface:

```bash
venv/bin/pip install -r requirements.txt
venv/bin/python app.py
```

Then open the local URL printed by Gradio in your browser.

`.env` is ignored by Git because it contains the API key.
