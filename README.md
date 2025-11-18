# Alfa Bank RAG System

Retrieval system for financial knowledge base queries. The system retrieves top-5 most relevant pages for user questions about Alfa Bank products.

**Test Score: Hit@5 = 0.65**

## Quick Start

### Install Dependencies
```bash
pip install sentence-transformers faiss-cpu rank-bm25 transformers torch
```

### Run Notebooks
```bash
# Baseline v1.0 (Hit@5 = 0.10)
jupyter notebook 01_baseline_gte_sentence_chunking.ipynb

# Baseline v2.0 (Hit@5 = 0.30)
jupyter notebook 02_baseline_improved_chunking.ipynb

# Final approach (Hit@5 = 0.65)
jupyter notebook 03_final_hybrid_search.ipynb
```

## Approach Evolution

| Version | Method | Score | Key Insight |
|---------|--------|-------|-------------|
| v1.0 | GTE + sentence chunking | 0.10 | Initial baseline |
| v2.0 | GTE + word chunking (350 words) | 0.30 | Better chunking = 3x improvement |
| **Final** | **BM25 + Query Expansion + Dense + CrossEncoder** | **0.65** | Hybrid approach = 6.5x improvement |

## Final Methodology

1. **Query Expansion** - Add domain synonyms (вклад → депозит, накопление...)
2. **BM25 Search** - Fast lexical matching (30% weight)
3. **Dense Embeddings** - Semantic matching with MPNet (70% weight)
4. **Hybrid Merge** - Combine both signals
5. **CrossEncoder Reranking** - Pair-wise semantic scoring for top-5

## Dataset

- **Input**: ~1,000 Russian financial queries + ~20,000 web pages
- **Output**: Top-5 page IDs per query in format `[id1, id2, id3, id4, id5]`
- **Metric**: Hit@5 (coverage of correct answer in top-5)

## Results

- **Baseline v1.0**: 0.10 (weak model + poor chunking)
- **Baseline v2.0**: 0.30 (improved chunking strategy)
- **Final**: 0.65 (hybrid search + advanced reranking)

## Technology Stack

- **Embeddings**: SentenceTransformer (MPNet)
- **Search**: FAISS (Dense) + BM25 (Lexical)
- **Reranking**: CrossEncoder (ms-marco-MiniLM)
- **Framework**: PyTorch, HuggingFace Transformers

## Files

- `01_baseline_gte_sentence_chunking.ipynb` - GTE with sentence splitting
- `02_baseline_improved_chunking.ipynb` - Improved word-level chunking
- `03_final_hybrid_search.ipynb` - BM25 + Dense + CrossEncoder
- `PROJECT.md` - Detailed report
- `submission_*.csv` - Results for each approach

## References

- [SentenceTransformer](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [rank-bm25](https://github.com/dorianbrown/rank_bm25)

## Hackathon

Alfa Bank RAG System Challenge | November 2025
