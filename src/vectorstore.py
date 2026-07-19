"""
Embed document chunks and build/load a FAISS vector store.
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.ingest import load_documents, split_documents

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_index")

# A small, fast, free embedding model. Swap for a bigger one if you want
# to compare retrieval quality (worth noting in your README).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vectorstore():
    """Build a fresh FAISS index from documents in data/ and save it to disk."""
    documents = load_documents()
    chunks = split_documents(documents)

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_DIR)

    print(f"Built index from {len(chunks)} chunks, saved to {INDEX_DIR}")
    return vectorstore


def load_vectorstore():
    """Load a previously built FAISS index from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    build_vectorstore()
