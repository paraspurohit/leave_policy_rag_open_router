# Model Selection Guide

Use this checklist before choosing a model for the RAG platform.

1. **Define the model's role**
   Decide whether the model is for embeddings, answer generation, reranking, or another task. An embedding model and a text-generation model are not interchangeable.

2. **Confirm the price**
   Filter for OpenRouter models with `$0` input and `$0` output pricing. Check whether the free variant has separate rate limits or availability restrictions.

3. **Check task suitability**
   Prefer generation models that support document question answering, retrieval-augmented generation, instruction following, and reliable refusal when information is missing.

4. **Check embedding suitability**
   Prefer embedding models designed for semantic search. The document chunks and user questions must use the same embedding model.

5. **Check context length**
   Ensure the generation model can accept the system prompt, user question, and all retrieved chunks together.

6. **Check required API features**
   Verify support for the endpoint and parameters we use, including embeddings, chat completions, temperature, and any structured output needed later.

7. **Check language and document quality**
   Test how well the model handles the document's language, tables, headings, numbers, dates, and policy exceptions.

8. **Measure grounded accuracy**
   Run the same evaluation questions against every candidate. Score factual correctness, page references, and whether the model avoids unsupported claims.

9. **Measure operational behavior**
   Compare latency, rate-limit errors, temporary unavailability, output length, and consistency across repeated questions.

10. **Choose a primary and fallback**
    Select the candidate with the best evaluation result, then keep a second compatible free model as a fallback. Record the model ID and evaluation date because free-model availability changes.

## Selection workflow

```text
List candidates
    ↓
Filter by role, price, context, and API support
    ↓
Test with the same document and questions
    ↓
Score accuracy, grounding, speed, and reliability
    ↓
Select primary model plus fallback
```

For this project, start with:

- Embeddings: `liquid/lfm-2.5-embedding-350m:free`
- Generation: `liquid/lfm-2.5-2.6b:free`

These are starting candidates, not permanent decisions. Confirm their current OpenRouter availability before each evaluation run.
