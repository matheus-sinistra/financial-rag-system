# Financial RAG System

> **[🚀 Live Demo](#)** · Built by [Matheus Ambrozio](https://github.com/matheus-sinistra)

A production-ready Retrieval-Augmented Generation (RAG) system for querying SEC 10-K annual reports and earnings call transcripts — with source citations and RAGAS evaluation.

---

## Problem

Financial analysts spend hours manually searching through hundreds of pages of 10-K filings and earnings call transcripts to answer specific questions. This system makes that instant and auditable.

## Approach

```
PDF Documents → Chunking → Embeddings → FAISS Vector Store
                                              ↓
User Question → Query Embedding → Top-K Retrieval → LLM → Answer + Sources
```

**Chunking strategy:** 512 tokens with 64-token overlap — balances context preservation with retrieval precision. Smaller chunks (256) lose context; larger (1024) reduce retrieval relevance.

## Results

| Metric | Score |
|---|---|
| Faithfulness | — |
| Context Precision | — |
| Answer Relevancy | — |
| Cost per 1000 queries (OpenAI) | — |
| Cost per 1000 queries (local) | $0.00 |

*Results populated after evaluation run.*

## Tech Stack

- **Embeddings:** `text-embedding-3-small` (OpenAI) · `all-MiniLM-L6-v2` (sentence-transformers)
- **Vector Store:** FAISS
- **LLM:** GPT-4o-mini
- **Orchestration:** LangChain
- **Evaluation:** RAGAS
- **UI:** Streamlit

## How to Run

```bash
git clone https://github.com/matheus-sinistra/financial-rag-system
cd financial-rag-system
pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Edit .env and add OPENAI_API_KEY=your_key_here

streamlit run app.py
```

## Project Structure

```
financial-rag-system/
├── data/              # 10-K PDFs (not committed — see data/README.md)
├── src/
│   ├── ingestion.py   # PDF parsing and chunking
│   ├── embeddings.py  # Embedding generation and FAISS indexing
│   ├── retrieval.py   # RAG pipeline
│   └── evaluation.py  # RAGAS metrics
├── notebooks/
│   └── 01_eda.ipynb   # Document analysis and chunking experiments
├── app.py             # Streamlit interface
└── requirements.txt
```

## What I Would Do With More Time

- Implement hybrid search (BM25 + dense retrieval)
- Add re-ranking with cross-encoder
- Expand to earnings call transcripts via Whisper transcription
- Fine-tune embedding model on financial domain corpus
- Add query routing for multi-document comparison questions
