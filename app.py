"""
app.py — Streamlit UI for Financial RAG System.
Run: streamlit run app.py
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

st.set_page_config(
    page_title="Financial RAG — 10-K Q&A",
    page_icon="📊",
    layout="wide",
)

# --- Cache expensive operations ---
@st.cache_resource(show_spinner="Loading FAISS index...")
def load_index():
    from src.embeddings import load_faiss_index
    return load_faiss_index()

# --- UI ---
st.title("📊 Financial RAG — 10-K Q&A")
st.caption("Ask questions about SEC 10-K annual reports from Apple, Microsoft, Alphabet, JPMorgan, Goldman Sachs")

st.markdown("""
> **Data:** 2025/2026 10-K filings from SEC EDGAR · **Model:** Groq llama-3.3-70b · **Embeddings:** BAAI/bge-small-en-v1.5
""")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieved passages (top-k)", min_value=3, max_value=10, value=5)
    company_filter = st.multiselect(
        "Filter by company",
        ["Alphabet", "Apple", "GoldmanSachs", "JPMorgan", "Microsoft"],
        default=[],
        help="Leave empty to search all companies"
    )
    
    st.divider()
    st.header("Example Questions")
    example_questions = [
        "What was Microsoft's total revenue in 2025?",
        "What are Goldman Sachs' main revenue segments?",
        "What is JPMorgan's CET1 capital ratio?",
        "How does Alphabet generate revenue?",
        "What are Apple's main business risks?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["question"] = q

# Load index
try:
    index, metadata = load_index()
except FileNotFoundError:
    st.error("FAISS index not found. Run `python -m src.embeddings` first to build the index.")
    st.stop()

# Question input
question = st.text_input(
    "Ask a question about the 10-K filings:",
    value=st.session_state.get("question", ""),
    placeholder="e.g. What was Microsoft's cloud revenue growth in 2025?",
    key="question_input"
)

if question:
    with st.spinner("Retrieving and generating answer..."):
        from src.retrieval import answer_question
        from src.embeddings import search
        
        # Apply company filter to metadata if set
        if company_filter:
            filtered_meta = [m for m in metadata if m.get("company") in company_filter]
            import faiss, numpy as np
            if filtered_meta:
                meta_to_use = filtered_meta
                idx_to_use = index  # Use full index, filter results post-retrieval
            else:
                meta_to_use = metadata
                idx_to_use = index
        else:
            meta_to_use = metadata
            idx_to_use = index
        
        result = answer_question(question, idx_to_use, meta_to_use, top_k=top_k)
    
    # Display answer
    st.markdown("### Answer")
    st.markdown(result["answer"])
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.caption(f"Model: {result['model']}")
    with col2:
        st.caption(f"Tokens used: {result.get('tokens_used', 'N/A')}")
    
    # Display sources
    if result["sources"]:
        st.markdown("### Retrieved Passages")
        for i, source in enumerate(result["sources"]):
            with st.expander(f"[{i+1}] {source['company']} — score: {source['score']:.3f}"):
                st.markdown(f"**Company:** {source['company']}")
                st.markdown(f"**Relevance score:** {source['score']:.3f}")
                st.markdown("**Passage:**")
                st.text(source["text"])
    else:
        st.info("No relevant passages found above the relevance threshold.")

# Footer
st.divider()
st.caption("Built by Matheus Ambrozio | [GitHub](https://github.com/matheus-sinistra/financial-rag-system)")
