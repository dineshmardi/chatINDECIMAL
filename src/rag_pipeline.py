import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import os
import time
import requests
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from ingest import prepare_chunks


# ------------------ ENV SETUP ------------------
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ------------------ EMBEDDING + INDEX ------------------
def build_index_and_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["content"] for chunk in chunks]

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index


# ------------------ RETRIEVAL ------------------
def retrieve_context(query, model, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    return [chunks[idx] for idx in indices[0]]


# ------------------ GENERATION (OPENROUTER) ------------------
def generate_answer(query, retrieved_chunks):
    """
    Real grounded LLM response using OpenRouter
    with strict hallucination control and error handling
    """

    if not OPENROUTER_API_KEY:
        return "OPENROUTER_API_KEY not found. Please set it in .env file."

    context = "\n\n".join(
        f"- {chunk['content']}" for chunk in retrieved_chunks
    )

    prompt = f"""
You are an AI assistant.
Answer the question ONLY using the context below.
Do NOT make assumptions or generalizations.
If the context does not clearly supported by the context the answer,
say exactly:
"Information not found in documents."

Context:
{context}

Question:
{query}

Answer:
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    # ---- Robust API handling (retry + graceful failure) ----
    for attempt in range(3):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except requests.exceptions.HTTPError as e:
            print(f"⚠️ OpenRouter HTTP error (attempt {attempt+1}/3): {e}")
            time.sleep(2)

        except requests.exceptions.RequestException as e:
            print(f"⚠️ Network error: {e}")
            return "Network error while contacting the language model."

        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            return "Unexpected error while generating the answer."

    return "The language model service is temporarily unavailable. Please try again later."


# ------------------ MAIN PIPELINE ------------------
if __name__ == "__main__":
    print("Starting RAG pipeline...\n")

    # Step 1: Prepare documents
    chunks = prepare_chunks()
    print(f"Total chunks: {len(chunks)}")

    # Step 2: Build FAISS index
    model, index = build_index_and_embeddings(chunks)
    print("FAISS index built\n")

    # Step 3: User input
    query = input("Ask your question: ").strip()

    if not query:
        print("No question asked. Exiting.")
        sys.exit(0)

    print(f"\nQuery: {query}\n")

    # Step 4: Retrieve context
    retrieved_chunks = retrieve_context(query, model, index, chunks)

    print("Retrieved Context:\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"--- Chunk {i} ({chunk['source']}) ---")
        print(chunk["content"][:300], "\n")

    # Step 5: Generate answer
    print("Calling OpenRouter LLM...\n")
    answer = generate_answer(query, retrieved_chunks)

    print("Final Answer:\n")
    print(answer)
