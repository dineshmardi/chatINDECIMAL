import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from sentence_transformers import SentenceTransformer
from ingest import prepare_chunks


def generate_embeddings(chunks):
    """
    Converts text chunks into vector embeddings
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings


if __name__ == "__main__":
    chunks = prepare_chunks()
    embeddings = generate_embeddings(chunks)

    print(f"Total embeddings created: {len(embeddings)}")
    print(f"Embedding vector length: {len(embeddings[0])}")
