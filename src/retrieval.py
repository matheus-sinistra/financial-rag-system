"""
retrieval.py — RAG pipeline: retrieve chunks + generate answer with Groq LLM.

Design decisions:
- top_k=5: better recall than 3, acceptable context window usage
- System prompt instructs LLM to cite sources explicitly
- If no relevant chunks found (score < threshold), returns "I don't know"
- Groq llama-3.3-70b: fast, free, context window 128K
"""

import os
from typing import List, Dict, Optional, Tuple

import faiss
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

from src.embeddings import search, load_faiss_index

GROQ_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MIN_RELEVANCE_SCORE = 0.4  # Below this, chunks are likely irrelevant

SYSTEM_PROMPT = """You are a financial analyst assistant with access to 10-K annual reports from major companies.

When answering questions:
1. Base your answer ONLY on the provided context passages
2. Always cite which company's 10-K you're drawing from
3. Include specific numbers, percentages, or dates when available
4. If the context doesn't contain enough information, say so clearly
5. Keep answers concise but complete (2-4 sentences typically)

Format: Answer first, then list sources as [Company, relevant detail]."""


def format_context(results: List[Dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(
            f"[Passage {i+1} — {r['company']} 10-K]\n{r['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def answer_question(
    question: str,
    index: faiss.Index,
    metadata: List[dict],
    top_k: int = 5,
    return_sources: bool = True,
) -> Dict:
    """
    Full RAG pipeline: retrieve → format → generate.
    
    Returns dict with:
        answer: str
        sources: list of retrieved passages with scores
        question: str
        model: str
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # 1. Retrieve relevant chunks
    results = search(question, index, metadata, top_k=top_k)
    
    # Filter low-relevance chunks
    filtered = [r for r in results if r["score"] >= MIN_RELEVANCE_SCORE]
    
    if not filtered:
        return {
            "answer": "I don't have enough relevant information in the 10-K filings to answer this question.",
            "sources": [],
            "question": question,
            "model": GROQ_MODEL,
        }
    
    # 2. Format context
    context = format_context(filtered)
    
    # 3. Generate answer
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context from 10-K filings:\n\n{context}\n\nQuestion: {question}"},
    ]
    
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.1,  # Low temp for factual financial Q&A
    )
    
    answer = response.choices[0].message.content.strip()
    
    return {
        "answer": answer,
        "sources": filtered if return_sources else [],
        "question": question,
        "model": GROQ_MODEL,
        "tokens_used": response.usage.total_tokens,
    }


def batch_answer(questions: List[str], index: faiss.Index, metadata: List[dict]) -> List[Dict]:
    """Answer multiple questions — used for RAGAS evaluation."""
    return [answer_question(q, index, metadata) for q in questions]


if __name__ == "__main__":
    print("Loading index...")
    index, metadata = load_faiss_index()
    
    test_questions = [
        "What was Microsoft's total revenue in the most recent fiscal year?",
        "How does Apple describe its main business risks?",
        "What is Goldman Sachs' primary source of revenue?",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = answer_question(q, index, metadata)
        print(f"A: {result['answer']}")
        print(f"Sources: {[s['company'] for s in result['sources']]}")
        print(f"Tokens: {result.get('tokens_used', 'N/A')}")
