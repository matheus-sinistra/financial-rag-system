"""
evaluation.py — RAGAS evaluation of the RAG pipeline.

Metrics:
- Faithfulness: Is the answer grounded in the retrieved context? (no hallucinations)
- Context Precision: Are the retrieved passages actually relevant to the question?
- Answer Relevancy: Does the answer address what was asked?

RAGAS uses an LLM-as-judge approach — we use Groq for this.
"""

import os
import json
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# RAGAS evaluation dataset format
EVAL_QUESTIONS = [
    {
        "question": "What was Microsoft's total revenue in fiscal year 2025?",
        "ground_truth": "Microsoft's total revenue in fiscal year 2025 was $279.0 billion."
    },
    {
        "question": "What are the main revenue segments of Goldman Sachs?",
        "ground_truth": "Goldman Sachs' main revenue segments are Global Banking & Markets and Asset & Wealth Management."
    },
    {
        "question": "What was Alphabet's net income in 2025?",
        "ground_truth": "Alphabet reported net income in their 2025 annual report."
    },
    {
        "question": "What is JPMorgan's CET1 capital ratio?",
        "ground_truth": "JPMorgan Chase reports its CET1 capital ratio in its annual filing."
    },
    {
        "question": "How much did Apple spend on research and development?",
        "ground_truth": "Apple discloses its R&D spending in its annual 10-K filing."
    },
    {
        "question": "What are Microsoft's main cloud computing products?",
        "ground_truth": "Microsoft's main cloud products include Azure, Microsoft 365, and Dynamics 365."
    },
    {
        "question": "What is Goldman Sachs' total assets?",
        "ground_truth": "Goldman Sachs reports its total assets in its balance sheet."
    },
    {
        "question": "What percentage of Alphabet revenue comes from advertising?",
        "ground_truth": "The majority of Alphabet's revenue comes from Google advertising."
    },
    {
        "question": "What is JPMorgan's total net revenue?",
        "ground_truth": "JPMorgan Chase reports its total net revenue in its income statement."
    },
    {
        "question": "What are the main risk factors for Microsoft's cloud business?",
        "ground_truth": "Microsoft describes competition, security, and regulatory risks for its cloud business."
    },
]


def run_ragas_evaluation(index, metadata) -> Dict:
    """
    Run RAGAS evaluation on the predefined question set.
    Uses Groq as the LLM judge.
    
    Returns dict with metric scores and per-question details.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from ragas.llms import LangchainLLMWrapper
        from langchain_groq import ChatGroq
        from datasets import Dataset
    except ImportError as e:
        print(f"RAGAS import error: {e}")
        return {"error": str(e)}
    
    from src.retrieval import answer_question
    
    print(f"Running RAGAS evaluation on {len(EVAL_QUESTIONS)} questions...")
    
    questions, answers, contexts, ground_truths = [], [], [], []
    
    for i, qa in enumerate(EVAL_QUESTIONS):
        print(f"  [{i+1}/{len(EVAL_QUESTIONS)}] {qa['question'][:60]}...")
        result = answer_question(qa["question"], index, metadata, top_k=5)
        
        questions.append(qa["question"])
        answers.append(result["answer"])
        contexts.append([s["text"] for s in result["sources"]])
        ground_truths.append(qa["ground_truth"])
    
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    # Use Groq as LLM judge
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )
    llm_wrapper = LangchainLLMWrapper(llm)
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm_wrapper,
    )
    
    scores = results.to_pandas()
    
    summary = {
        "faithfulness": float(scores["faithfulness"].mean()),
        "answer_relevancy": float(scores["answer_relevancy"].mean()),
        "context_precision": float(scores["context_precision"].mean()),
        "n_questions": len(EVAL_QUESTIONS),
        "per_question": scores[["question", "faithfulness", "answer_relevancy", "context_precision"]].to_dict("records"),
    }
    
    return summary


def print_ragas_report(summary: Dict):
    """Pretty-print RAGAS results."""
    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness:       {summary['faithfulness']:.3f}  (no hallucinations)")
    print(f"Answer Relevancy:   {summary['answer_relevancy']:.3f}  (on-topic answers)")
    print(f"Context Precision:  {summary['context_precision']:.3f}  (retrieval quality)")
    print(f"\nBased on {summary['n_questions']} questions from 5 company 10-Ks")
    
    avg = (summary['faithfulness'] + summary['answer_relevancy'] + summary['context_precision']) / 3
    print(f"\nOverall score:      {avg:.3f}")
    
    if avg >= 0.75:
        print("Rating: ✅ Good (portfolio-ready)")
    elif avg >= 0.60:
        print("Rating: ⚠️  Acceptable (could improve chunking or retrieval)")
    else:
        print("Rating: ❌ Needs improvement")


if __name__ == "__main__":
    from src.embeddings import load_faiss_index
    
    print("Loading index for RAGAS evaluation...")
    index, metadata = load_faiss_index()
    
    summary = run_ragas_evaluation(index, metadata)
    
    if "error" not in summary:
        print_ragas_report(summary)
        
        # Save results to disk
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/ragas_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print("\nResults saved → data/processed/ragas_results.json")
    else:
        print(f"Evaluation failed: {summary['error']}")
