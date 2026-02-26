"""
embeddings.py — Generate BGE embeddings and build FAISS index.

Model: BAAI/bge-small-en-v1.5
- 384 dimensions, ~33M params
- Designed for retrieval (not just similarity)
- BGE prefix instruction: "Represent this sentence for searching relevant passages:"
- Free, no API key, runs on CPU in <1s per batch
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str], is_query: bool = False, batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of strings.
    BGE models use a query prefix for better retrieval — applied when is_query=True.
    """
    model = get_model()
    
    if is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,  # L2 norm → cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


def build_faiss_index(chunks: List[Document], index_path: str = "data/processed/index.faiss",
                      meta_path: str = "data/processed/metadata.pkl") -> faiss.Index:
    """
    Embed all chunks and build a FAISS flat L2 index.
    Saves index and metadata to disk.
    
    FAISS FlatIP (inner product) with normalized vectors = cosine similarity.
    No approximation (exact search) — 6K chunks is small enough for exact search.
    For 1M+ chunks, switch to IVFFlat or HNSW.
    """
    os.makedirs("data/processed", exist_ok=True)
    
    texts = [c.page_content for c in chunks]
    print(f"Embedding {len(texts)} chunks with {EMBEDDING_MODEL}...")
    
    embeddings = embed_texts(texts, is_query=False, batch_size=64)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vecs)
    index.add(embeddings)
    
    faiss.write_index(index, index_path)
    
    metadata = [c.metadata | {"text": c.page_content} for c in chunks]
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"  Index: {index.ntotal} vectors, dim={dim}")
    print(f"  Saved → {index_path}")
    print(f"  Metadata → {meta_path}")
    
    return index


def load_faiss_index(index_path: str = "data/processed/index.faiss",
                     meta_path: str = "data/processed/metadata.pkl"):
    """Load pre-built FAISS index and metadata from disk."""
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def search(query: str, index: faiss.Index, metadata: List[dict], top_k: int = 5) -> List[dict]:
    """
    Retrieve top-k chunks for a query.
    Returns list of dicts with text, company, score, chunk_id.
    """
    query_vec = embed_texts([query], is_query=True)
    scores, indices = index.search(query_vec, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        meta = metadata[idx].copy()
        meta["score"] = float(score)
        results.append(meta)
    
    return results


if __name__ == "__main__":
    from src.ingestion import load_all_documents, chunk_documents
    
    print("Building FAISS index from 10-K documents...")
    docs = load_all_documents("data/raw")
    chunks = chunk_documents(docs)
    
    index = build_faiss_index(chunks)
    
    # Quick smoke test
    print("\nSmoke test — query: 'What is Microsoft revenue?'")
    index, metadata = load_faiss_index()
    results = search("What is Microsoft revenue?", index, metadata, top_k=3)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r['company']} | score={r['score']:.3f}")
        print(f"     {r['text'][:150]}...")
