"""
Build the retrieval-augmented generation chain: retriever + prompt + LLM.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.vectorstore import load_vectorstore

load_dotenv()

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based only on the
provided context. If the answer isn't in the context, say you don't know —
do not make up information.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(k: int = 4):
    """Assemble the full RAG chain: retriever -> prompt -> LLM -> parser."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})


    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


if __name__ == "__main__":
    chain, retriever = build_rag_chain()

    question = "What is this document about?"
    answer = chain.invoke(question)

    print(f"Q: {question}")
    print(f"A: {answer}")
