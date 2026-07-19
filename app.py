"""
Streamlit front end for the RAG Document Q&A Assistant.

Run with: streamlit run app.py
"""

import os
import streamlit as st

from src.vectorstore import build_vectorstore
from src.rag_chain import build_rag_chain

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 RAG Document Q&A Assistant")
st.caption("Ask questions grounded in your own documents.")

INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")

with st.sidebar:
    st.header("Setup")
    st.write("Documents in `data/` are indexed on first run.")
    if st.button("Rebuild index"):
        with st.spinner("Building index..."):
            build_vectorstore()
        st.success("Index rebuilt.")

if not os.path.exists(INDEX_DIR):
    st.info("No index found yet. Click **Rebuild index** in the sidebar to get started.")
    st.stop()

if "chain" not in st.session_state:
    st.session_state.chain, st.session_state.retriever = build_rag_chain()

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your documents...")

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = st.session_state.chain.invoke(question)
        st.write(answer)

        # Show retrieved sources so users (and interviewers) can see grounding
        with st.expander("Sources used"):
            docs = st.session_state.retriever.invoke(question)
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}**")
                st.text(doc.page_content[:300] + "...")

    st.session_state.history.append((question, answer))
