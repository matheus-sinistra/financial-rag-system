"""
ingestion.py — Load 10-K HTML filings and split into chunks for RAG.

Chunking strategy: 512 tokens / 64 overlap
- 512 tokens: enough context for financial statements and MD&A paragraphs
- 64 overlap: preserves cross-chunk sentence boundaries
- Smaller (256) loses context for multi-sentence answers
- Larger (1024) degrades retrieval precision
"""

import os
import re
from pathlib import Path
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def _clean_html(html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_document(filepath: str) -> str:
    """Load a 10-K HTML file and return clean text."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    return _clean_html(html)


def load_all_documents(data_dir: str = "data/raw") -> List[Dict]:
    """
    Load all 10-K HTML files from data_dir.
    Returns list of dicts with keys: company, filepath, text, char_count
    """
    docs = []
    data_path = Path(data_dir)
    
    for html_file in sorted(data_path.glob("*.html")):
        company = html_file.stem.replace("_10K_2025", "")
        text = load_document(str(html_file))
        docs.append({
            "company": company,
            "filepath": str(html_file),
            "text": text,
            "char_count": len(text),
        })
        print(f"  Loaded {company}: {len(text):,} chars")
    
    return docs


def chunk_documents(
    docs: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Document]:
    """
    Split documents into LangChain Document chunks with metadata.
    
    Args:
        docs: Output of load_all_documents()
        chunk_size: Target chunk size in characters (≈ tokens for English)
        chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        List of LangChain Document objects with metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    all_chunks: List[Document] = []
    
    for doc in docs:
        raw_chunks = splitter.split_text(doc["text"])
        
        for idx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < 50:  # Skip near-empty chunks
                continue
            
            all_chunks.append(Document(
                page_content=chunk_text,
                metadata={
                    "company": doc["company"],
                    "source": doc["filepath"],
                    "chunk_id": idx,
                    "total_chunks": len(raw_chunks),
                }
            ))
    
    return all_chunks


def get_ingestion_stats(docs: List[Dict], chunks: List[Document]) -> Dict:
    """Return summary stats for logging / README."""
    stats = {
        "documents": len(docs),
        "total_chars": sum(d["char_count"] for d in docs),
        "total_chunks": len(chunks),
        "avg_chunk_size": sum(len(c.page_content) for c in chunks) // max(len(chunks), 1),
        "companies": [d["company"] for d in docs],
    }
    return stats


if __name__ == "__main__":
    print("Loading 10-K documents...")
    docs = load_all_documents("data/raw")
    
    print("\nChunking (512 tokens / 64 overlap)...")
    chunks = chunk_documents(docs)
    
    stats = get_ingestion_stats(docs, chunks)
    print(f"\n=== Ingestion Stats ===")
    print(f"Documents:    {stats['documents']}")
    print(f"Total chars:  {stats['total_chars']:,}")
    print(f"Total chunks: {stats['total_chunks']:,}")
    print(f"Avg chunk:    {stats['avg_chunk_size']} chars")
    print(f"Companies:    {', '.join(stats['companies'])}")
