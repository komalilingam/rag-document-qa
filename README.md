# 📚 RAG Document Question Answering System

A Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask natural language questions. The system retrieves the most relevant document chunks using semantic search and generates context-aware answers using Google's Gemini API.

---

## 🚀 Project Overview

Large Language Models have extensive general knowledge but cannot answer questions about private or custom documents without additional context.

This project solves that problem using Retrieval-Augmented Generation (RAG). Instead of relying solely on the LLM, the application retrieves relevant document chunks from a vector database and supplies them to Gemini before generating the final answer.

The application provides accurate, context-grounded responses while reducing hallucinations.

---

## ✨ Features

- Upload PDF and TXT documents
- Automatic document chunking
- Semantic search using vector embeddings
- FAISS vector database for efficient retrieval
- Google Gemini integration for answer generation
- Streamlit-based interactive web interface
- Modular Python project structure
- Evaluation support using RAGAS

---

## 🏗️ System Architecture

```
Documents
     │
     ▼
Document Loader
     │
     ▼
Text Splitter
     │
     ▼
Embeddings
     │
     ▼
FAISS Vector Store
     │
User Question
     │
     ▼
Similarity Search
     │
Relevant Context
     │
     ▼
Gemini LLM
     │
     ▼
Generated Answer
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Framework | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Generative AI Embeddings |
| Vector Database | FAISS |
| Framework | LangChain |
| Evaluation | RAGAS |
| Environment | Python Virtual Environment |

---

## 📂 Project Structure

```
rag-qa-project/

│── app.py
│── requirements.txt
│── README.md
│── .gitignore
│── .env

├── data/
│     ├── sample1.pdf
│     ├── sample2.txt

├── vectorstore/

├── src/
│     ├── loader.py
│     ├── splitter.py
│     ├── embeddings.py
│     ├── vectorstore.py
│     ├── rag_pipeline.py
│     └── evaluate.py

└── screenshots/
      └── app.png
```

---

## ⚙️ Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/rag-qa-project.git
```

Move into the project directory.

```bash
cd rag-qa-project
```

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file.

```
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

---

## ▶️ Running the Project

Generate the vector database.

```bash
python -m src.vectorstore
```

Start the Streamlit application.

```bash
streamlit run app.py
```

---

## 📷 Application Screenshot

Add a screenshot after running the application.

```
<img width="1894" height="919" alt="image" src="https://github.com/user-attachments/assets/2bb73790-61df-40ba-b737-c208ce9cc51a" />

```

Example:

```md
![Application Screenshot](screenshots/app.png)
```

---

## 📊 Evaluation

The project includes an evaluation module using **RAGAS** to assess retrieval and answer quality.

Metrics include:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall

### Current Status

| Metric | Score |
|---------|-------|
| Faithfulness | Pending Evaluation |
| Answer Relevancy | Pending Evaluation |
| Context Precision | Pending Evaluation |
| Context Recall | Pending Evaluation |

Run the evaluation using:

```bash
python -m src.evaluate
```

---

## 📄 Supported File Types

- PDF
- TXT

---

## 🎯 Use Cases

- Company knowledge base
- Academic document search
- Research paper Q&A
- Resume analysis
- Personal document assistant
- Technical documentation search

---

## ⚠️ Known Limitations

- Supports only PDF and TXT documents
- No multi-turn conversation memory
- Retrieval quality depends on chunk size
- Uses a single embedding model
- Large documents may require additional optimization

---

## 🚀 Future Improvements

- Multi-document collections
- Chat history support
- Hybrid keyword + semantic retrieval
- Re-ranking retrieved chunks
- OCR support for scanned PDFs
- Support for DOCX, PPTX, and HTML documents
- Docker deployment
- Cloud deployment on Render or AWS
- Authentication for multiple users

---

## 📌 Skills Demonstrated

- Retrieval-Augmented Generation (RAG)
- Large Language Models
- LangChain
- FAISS
- Vector Embeddings
- Semantic Search
- Prompt Engineering
- Streamlit
- Python Development
- API Integration
- RAG Evaluation using RAGAS

---

## 👨‍💻 Author

**Komali lingam**

AI & Data Science Student

GitHub:https://github.com/komalilingam



---

## ⭐ If you found this project useful, consider giving it a star!
