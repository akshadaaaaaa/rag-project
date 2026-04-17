# Document Q&A — RAG System

A Retrieval Augmented Generation (RAG) system that lets you chat with any
document. Ask questions and get accurate answers grounded in the document :
not from the LLM's training data.

## What it does

- Load any text document
- Split it into chunks, embed each chunk using HuggingFace
- Store embeddings in ChromaDB (local vector database)
- At query time: embed the question, find the 3 most relevant chunks,
  inject them into the prompt, and generate a grounded answer
- Shows which parts of the document were used to answer

## Why RAG matters

Without RAG, LLMs answer from training data and hallucinate.
With RAG, answers are grounded in your actual documents — verifiable,
accurate, and auditable. Critical for enterprise and regulated environments.

## Tech stack

- **Python** — core language
- **sentence-transformers** — HuggingFace model (all-MiniLM-L6-v2)
  converts text into 384-dimensional embedding vectors
- **ChromaDB** — local vector database stores and searches embeddings
- **Groq API** — free LLM inference (Llama 3.3 70B) generates the answer
- **FastAPI** — REST API wrapper with chat UI
- **python-dotenv** — keeps API key secure

## How to run

**1. Install dependencies**
```bash
pip install sentence-transformers chromadb groq fastapi uvicorn python-dotenv
```

**2. Create a .env file**
```
GROQ_API_KEY=your-groq-key-here
```

**3. Add your document**

Put any text file in the project folder and name it `document.txt`

**4. Run**
```bash
python rag.py
```

**5. Open browser**
```
http://localhost:8000
```

## The RAG pipeline

```
INDEXING (once):
Document → Chunk (500 chars) → Embed (384 vectors) → Store in ChromaDB

QUERY (every question):
Question → Embed → Search ChromaDB → Top 3 chunks
→ Build prompt → Groq LLM → Grounded answer + sources
```

## The four core functions

| Function | What it does |
|---|---|
| `load_document()` | Reads the text file |
| `chunk_text()` | Splits into 500-char pieces with 50-char overlap |
| `embed_and_store()` | HuggingFace embeddings → ChromaDB |
| `ask()` | Retrieves relevant chunks → builds prompt → calls LLM |

## Example questions (using a CV as document)

- "What are the candidate's key skills?"
- "Where did they study?"
- "What did they achieve at their last role?"

## What I learned building this

- How embeddings represent meaning as vectors
- How vector similarity search works (semantic not keyword)
- How chunking strategy affects retrieval quality
- How to ground LLM answers in verified sources
- How to prevent hallucination using prompt constraints
- How to wrap a Python AI pipeline in a REST API


