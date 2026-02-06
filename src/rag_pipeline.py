import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from ingest import prepare_chunks


def build_index_and_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts)

    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return model, index


def retrieve_context(query, model, index, chunks, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = []
    for idx in indices[0]:
        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


def generate_answer(query, retrieved_chunks):
    """
    Simulated grounded LLM response.
    In real deployment, replace this with OpenRouter / Ollama call.
    """

    context = "\n\n".join(
        f"- {chunk['content']}" for chunk in retrieved_chunks
    )

    prompt = f"""
You are an AI assistant.
Answer the question ONLY using the context below.
If the answer is not present, say: "Information not found in documents."

Context:
{context}

Question:
{query}

Answer:
"""

    # Simulated generation (placeholder logic)
    if "escrow" in context.lower():
        return (
            "Indecimal ensures payment safety through an escrow-based payment model, "
            "where customer payments are verified at each construction stage before "
            "being released to the construction partner."
        )
    else:
        return "Information not found in documents."


if __name__ == "__main__":
    chunks = prepare_chunks()
    model, index = build_index_and_embeddings(chunks)

    query = "How does Indecimal ensure payment safety?"
    retrieved_chunks = retrieve_context(query, model, index, chunks)

    print("\nRetrieved Context:\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"--- Chunk {i} ({chunk['source']}) ---")
        print(chunk["content"][:300], "\n")

    answer = generate_answer(query, retrieved_chunks)

    print("\nFinal Answer:\n")
    print(answer)
