import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from ingest import prepare_chunks


def build_faiss_index(embeddings):
    """
    Build a FAISS index from embeddings
    """
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index


def retrieve_top_k(query, model, index, chunks, k=3):
    """
    Retrieve top-k most relevant chunks for a query
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results


if __name__ == "__main__":
    # Prepare data
    chunks = prepare_chunks()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts)

    embeddings = np.array(embeddings).astype("float32")

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Test query
    query = "How does Indecimal ensure payment safety?"
    retrieved_chunks = retrieve_top_k(query, model, index, chunks)

    print("\nQuery:", query)
    print("\nRetrieved Chunks:\n")

    for i, chunk in enumerate(retrieved_chunks, 1):
        print(f"--- Chunk {i} (Source: {chunk['source']}) ---")
        print(chunk["content"][:300], "\n")
