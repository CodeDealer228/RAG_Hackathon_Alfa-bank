# RAG System for Alfa Bank - Project Report

## Overview

This project addresses the retrieval-augmented generation (RAG) problem for Alfa Bank's financial knowledge base. The task is to build a system that retrieves the top-5 most relevant web pages for a given user query.

## Problem Statement

Given:
- ~1,000 financial queries in Russian
- ~20,000 pages from Alfa Bank's website
- Each page contains information about financial products (deposits, loans, investments, cards, insurance, etc.)

Goal: For each query, return a ranked list of 5 most relevant page IDs

Metric: Hit@5 (percentage of queries where the correct answer appears in top-5)

## Dataset Specifications

### Input Data

**questions_clean.csv**
- Queries in Russian with spelling errors, emojis, and colloquial expressions
- Example: "Здравствуйте! Пожалуйста, расскажи про вклады 😊"

**websites_update.csv**
- Page ID, title, and full text content
- Mixed text with HTML fragments, emojis, and special characters (\xa0, \n)

### Target Format

CSV file with columns:
- `q_id`: Query identifier
- `web_list`: List of 5 web page IDs in format `[id1, id2, id3, id4, id5]`

## Experimental Results

| Approach | Method | Hit@5 | Notes |
|----------|--------|-------|-------|
| Baseline v1.0 | GTE + sentence-level chunking | 0.10 | Weak embeddings, poor chunking strategy |
| Baseline v2.0 | GTE + word-level chunking (350 words, 80% overlap) | 0.30 | 3x improvement through better chunking |
| **Final** | **BM25 + Query Expansion + Hybrid Search + CrossEncoder** | **0.65** | 6.5x improvement vs baseline |

## Methodology

### Baseline Approach (Hit@5 = 0.10-0.30)

**v1.0: Simple Sentence-Level Chunking**
- Split documents by sentence boundaries
- Use GTE embeddings (Alibaba-NLP/gte-multilingual-base)
- FAISS L2 index for nearest neighbor search
- Result: Poor performance due to weak model and fragmented chunks

**v2.0: Improved Word-Level Chunking**
- Chunk documents into 350-word segments with 80% overlap
- Maintain full preprocessing (emoji removal, normalization)
- Same embedding model and search strategy
- Result: Hit@5 = 0.30 (3x improvement)

### Final Approach (Hit@5 = 0.65)

**Component 1: Query Expansion**
- Expand user queries with financial domain synonyms
- Dictionary mapping: "вклад" → [вклад, депозит, накопление, сбережение]
- Improves lexical search coverage by 15-25%

**Component 2: BM25 Lexical Search**
- Build BM25 index on preprocessed documents
- Fast keyword-based ranking
- Normalized scores: 30% weight in hybrid search

**Component 3: Dense Embeddings + FAISS**
- Use MPNet embeddings (sentence-transformers/paraphrase-multilingual-mpnet-base-v2)
- FAISS IndexFlatIP for efficient similarity search
- Normalized scores: 70% weight in hybrid search
- Captures semantic relevance

**Component 4: Hybrid Search Combination**
- Merge BM25 (lexical) and Dense (semantic) results
- Formula: `combined_score = 0.3 * bm25_norm + 0.7 * dense_norm`
- Produces top-100 candidates combining both signal types

**Component 5: CrossEncoder Reranking**
- Fine-tuned pairwise ranking model (ms-marco-MiniLM-L-6-v2)
- Evaluates (query, candidate_document) pairs
- Final reranking to top-5 results
- Adds 10-15% improvement through precision reranking

## Why This Works

1. **Query Expansion**: Captures domain-specific terminology and synonyms
2. **Lexical + Semantic Fusion**: BM25 catches keyword matches, Dense embeddings understand meaning
3. **CrossEncoder Precision**: Final reranking based on semantic pair-wise relevance
4. **Weighted Combination**: 70% dense (semantic) + 30% lexical balances coverage and relevance

## Failed Approaches

- **Query question extraction**: Attempted to identify FAQ-style questions in documents - no improvement
- **Heavy text preprocessing**: Over-aggressive cleaning and stopword removal - reduced recall
- **Complex chunking strategies**: Sophisticated NLP-based chunking on limited compute - marginal gains vs overhead

## Running the Code

### Requirements
```
sentence-transformers>=2.2
faiss-cpu>=1.7
rank-bm25>=0.2
transformers>=4.30
```

### Execution
```bash
# Baseline v1 (Hit@5 = 0.10)
jupyter notebook 01_baseline_gte_sentence_chunking.ipynb

# Baseline v2 (Hit@5 = 0.30)
jupyter notebook 02_baseline_improved_chunking.ipynb

# Final approach (Hit@5 = 0.65)
jupyter notebook 03_final_hybrid_search.ipynb
```

## File Structure

```
rag-alfa-bank/
├── README.md
├── PROJECT.md (this file)
├── 01_baseline_gte_sentence_chunking.ipynb      (Hit@5 = 0.10)
├── 02_baseline_improved_chunking.ipynb          (Hit@5 = 0.30)
├── 03_final_hybrid_search.ipynb                 (Hit@5 = 0.65)
├── submission_baseline_v1.csv
├── submission_baseline_v2.csv
└── submission_final.csv
```

## Key Learnings

1. Chunking strategy is critical - word-level with overlap > document-level
2. Domain-aware query expansion provides consistent 15-25% boost
3. Hybrid search (lexical + semantic) more robust than pure semantic
4. CrossEncoder provides meaningful but diminishing returns in reranking
5. Proper text preprocessing (emoji removal, normalization) matters for embedding quality

## Time Complexity

- BM25 indexing: O(n*m) where n=documents, m=avg_words_per_doc
- FAISS search: O(n*d) where n=documents, d=embedding_dim
- CrossEncoder reranking: O(k*seq_len²) where k=top_k_candidates
- Total pipeline on GPU T4: ~15-20 minutes for 20K documents

## References

- SentenceTransformer: https://github.com/UKPLab/sentence-transformers
- FAISS: https://github.com/facebookresearch/faiss
- BM25: https://github.com/dorianbrown/rank_bm25
- CrossEncoder: https://www.sbert.net/docs/cross-encoders/models/models.html

## Contact

Hackathon: Alfa Bank RAG System Challenge (November 2025)
