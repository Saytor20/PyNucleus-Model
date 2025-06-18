# Source Citation Standard

## Overview

This document defines the citation standard for the PyNucleus RAG (Retrieval-Augmented Generation) system to ensure factual accuracy and traceability of generated answers.

## Citation Format

### JSON Citation Structure

Each citation must include the following fields:

```json
{
  "source_filename": "<original_document>.json",
  "chunk_id": "<source_id>_<chunk_idx>",
  "similarity": 0.87
}
```

### Field Definitions

- **source_filename**: Original document filename from which the chunk was extracted
- **chunk_id**: Unique identifier combining source ID and chunk index
- **similarity**: Similarity score between query and chunk (0.0 to 1.0)

### Inline Citation Markers

Answers must include inline citation markers in the format `[†1]`, `[†2]`, etc., mapping to the citation objects by index order.

Example:
```
Modular chemical plants offer several key advantages [†1]. The design principles focus on standardization and scalability [†2].
```

## Implementation Requirements

### 1. Citation Tracking
- All retrieved chunks must be tracked with metadata
- Source filename and chunk ID must be preserved through processing
- Similarity scores must be rounded to 4 decimal places

### 2. Answer Generation
- Inline citation markers must map sequentially to citation list
- Each factual claim should reference at least one source
- Multiple citations can support the same claim

### 3. Logging Requirements
- All queries and responses logged to `logs/rag_trace.jsonl`
- Each log entry must include:
  - `query_id`: Unique identifier for the query
  - `timestamp`: ISO format timestamp
  - `question`: Original user question
  - `answer`: Generated answer with citation markers
  - `citations`: Array of citation objects
  - `metadata`: Query processing metadata

### 4. Validation Criteria
- Word-level overlap ≥ 85% between ideal answer and LLM answer must appear in cited chunks
- All citation objects must have required fields
- Citation markers must sequentially reference citation array

## Code References

### Pipeline Integration
- `src/pynucleus/pipeline/pipeline_rag.py:query_with_citations()` - Main query method
- `src/pynucleus/pipeline/pipeline_rag.py:_generate_answer_with_citations()` - Answer generation
- `src/pynucleus/pipeline/pipeline_rag.py:_log_citation_trace()` - Citation logging

### Validation Components
- `scripts/validate_rag_factual_accuracy.py` - Factual accuracy validation
- `src/pynucleus/tests/test_rag_citations.py` - Unit tests for citation structure

## Quality Gates

1. **Structure Validation**: All citations must have required fields
2. **Factual Accuracy**: ≥90% of content must be verifiable in cited sources
3. **Traceability**: All answers must be traceable to specific document chunks
4. **Logging Completeness**: All queries must be logged with full metadata

## Example Usage

```python
from src.pynucleus.pipeline.pipeline_rag import RAGPipeline

pipeline = RAGPipeline()
response = pipeline.query_with_citations(
    "What are the benefits of modular chemical plants?",
    k=5,
    similarity_threshold=0.3
)

print(f"Answer: {response['answer']}")
print(f"Citations: {len(response['citations'])}")
```

## Benchmark Targets

- **Factual Accuracy**: ≥90% (configurable in `docs/benchmark_target.md`)
- **Citation Coverage**: 100% of factual claims must be cited
- **Response Time**: <5 seconds per query
- **Similarity Threshold**: ≥0.1 for inclusion in citations 