"""
Load documents from the data/ folder and split them into chunks
ready for embedding.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# TODO: experiment with these and document your findings in the README
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents(data_dir: str = DATA_DIR):
    """Load all PDF and TXT files from data_dir into LangChain Document objects."""
    documents = []

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())
        elif filename.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
        # TODO: add loaders for .docx, .csv, .md etc. as needed

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks for better retrieval granularity."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} raw documents")

    chunks = split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    if chunks:
        print("\n--- Sample chunk ---")
        print(chunks[0].page_content[:300])
