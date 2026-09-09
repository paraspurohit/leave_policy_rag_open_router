# RAG App Notes

## What this app does

This app answers questions from the leave-policy PDF. It uses RAG, which means **Retrieval-Augmented Generation**.

The app first finds relevant parts of the PDF, then gives those parts to an AI model to write the answer.

## Main flow

```text
PDF
  ↓
Extract text
  ↓
Split text into chunks
  ↓
Convert chunks into embedding vectors
  ↓
Store vectors in FAISS
  ↓
Convert user question into a vector
  ↓
Find similar chunks
  ↓
Send chunks to the generation model
  ↓
Return an answer with page references
```

## Important terms

- **PDF extraction:** Reads text and page numbers from the PDF.
- **Chunk:** A smaller piece of the PDF text.
- **Embedding:** A numerical vector representing the meaning of text.
- **Vector store:** A searchable collection of embedding vectors.
- **Retrieval:** Finding chunks that are similar to the user question.
- **Generation:** Writing the final answer using the retrieved chunks.
- **RAG:** Retrieval plus generation.

## Models

- Embedding model: `liquid/lfm-2.5-embedding-350m:free`
- Generation model: `liquid/lfm-2.5-2.6b:free`

Both are configured through OpenRouter and use free model variants. Free models can have rate limits, temporary availability issues, and changing model IDs.

The embedding model is used only for vectors. The generation model is used only for writing answers.

## Important files

- `app.py`: Gradio web interface.
- `test.py`: Terminal-based question interface.
- `src/pdf_loader.py`: Extracts PDF text and page numbers.
- `src/chunker.py`: Creates page-aware chunks.
- `src/embeddings.py`: Calls the embedding model.
- `src/vector_store.py`: Stores and searches vectors with FAISS.
- `src/retriever.py`: Finds relevant chunks for a question.
- `src/generator.py`: Generates a grounded answer.
- `src/rag_pipeline.py`: Connects the complete RAG workflow.
- `src/model_selector.py`: Stores model candidates and selection logic.
- `evaluation/questions.json`: Test questions and expected facts.
- `evaluation/evaluate_models.py`: Compares generation candidates.
- `evaluation/results.json`: Saved evaluation results.
- `.env`: API key and model configuration. Never share this file.

## Chunking note

The embedding model accepts a maximum of 512 tokens per input. The app therefore uses a default chunk size of 1,200 characters to leave a safety margin.

## Model evaluation

The correct model should be selected by testing all candidates with the same questions. Check:

1. Correctness
2. Page references
3. Use of only retrieved context
4. Ability to say when information is missing
5. Response speed
6. API reliability

The current successful candidate is `liquid/lfm-2.5-2.6b:free`. Some other candidates failed because they were restricted or had invalid IDs.

## Running the app

Install dependencies:

```bash
venv/bin/pip install -r requirements.txt
```

Run the terminal version:

```bash
venv/bin/python test.py
```

Run the Gradio version:

```bash
venv/bin/python app.py
```

Run tests:

```bash
venv/bin/pytest -q
```

Run model evaluation:

```bash
venv/bin/python evaluation/evaluate_models.py
```

## Gradio Flag button

The Flag button saves a question and answer that a user marks as useful or incorrect. These saved examples can later help improve evaluation questions, prompts, retrieval, or model selection.

## Common errors

- **Model does not exist:** Check the current OpenRouter model ID.
- **Embedding input exceeds 512 tokens:** Reduce the chunk size.
- **Connection error:** Check internet or DNS access.
- **Missing API key:** Check `.env`.
- **`fitz` deprecation warning:** It is a warning, not a pipeline failure.
