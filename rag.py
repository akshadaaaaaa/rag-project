import os
from groq import Groq
from dotenv import load_dotenv
import chromadb
from chromadb import Client
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

#Loading API key
load_dotenv()
client = Groq(api_key = os.getenv("GROQ_API_KEY"))

#Load the embedding model - this turns text into vectors
#all-MiniLM-L6-v2 is small, fast, and free
embedder = SentenceTransformer("all-MiniLM-L6-v2")

#Set up ChromaDB- local vector database which stores your
#document chunks as vectors on your machine
chroma = Client()
collection = chroma.create_collection("my_documents")

#print("Setup done")

#loading the document
def load_document(filepath):
    with open(filepath,"r", encoding="utf-8") as f:
        text = f.read()
    print(f"Loaded document: {len(text)} characters")
    return text

#Chunking -splitting document into chunks
#chunk size = how many characters per chunk
#overlap = how many characters to repeat between chunks
#overlap prevents important content from being cut off at the boundary

def chunk_text(text, chunk_size =500, overlap= 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    print(f"Created {len(chunks)} chunks")
    return chunks


#Embedding and storing in ChromaDB
# embed each chunk and store in ChromaDB
#This is the indexing steo - runs once when you load your document

def embed_and_store(chunks):
    for i, chunk in enumerate(chunks):
        #turn every chunk text into vector using HUggingface
        embedding = embedder.encode(chunk).tolist()

        #Store in ChromaDB with :
        #embedding = the vector
        #document = the original text(so we can return it later)
        #id = unique identifier for each chunk

        collection.add(
            embeddings=[embedding],
            documents =[chunk],
            ids=[f"chunk_{i}"]
        )
    print(f"Stored {len(chunks)} chunks in ChromaDB")


#Asking Qs and getting the answer

def ask(question):
    #Step 1: embed the question (same way as we embedded chunks)
    question_embedding = embedder.encode(question).tolist()

    #Step 2: search ChromaDB for the 3 most similar chunks
    results = collection.query(
            query_embeddings = [question_embedding],
            n_results = 3
    )

    #Step 3: pull out the actual text of those chunks

    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    #Step 4: build the prompt
    #system prompt tell LLM to ONLY use the retrieved context
    #this prevents hallucination

    prompt = f"""Answer the question using ONLY the context below.
    If the answer is not in the context, say "I don't have that information."

    Context:
    {context}

    Question: {question}"""

    #Step 5: Call GROQ LLM with Prompt
    response = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role": "user","content": prompt}],
        max_tokens = 512
    )

    answer = response.choices[0].message.content
    return answer, retrieved_chunks

#Ask a question
# question = "When did the EU AI act commence?"
# answer, sources = ask(question)

# print(f"\nQuestion: {question}")
# print(f"\nAnswer: {answer}")
# print(f"\nRetrieved from these chunks: ")
# for i, chunk in enumerate(sources):
#     print(f"\nChunk {i+1}: {chunk[:100]}...")

# Load document on startup
text = load_document("document.txt")
chunks = chunk_text(text)
embed_and_store(chunks)
print("RAG system ready.")

# FastAPI app
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/ask")
def ask_endpoint(request: ChatRequest):
    answer, sources = ask(request.message)
    return {
        "answer": answer,
        "sources": [s[:150] for s in sources]
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return open("ui.html").read()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
