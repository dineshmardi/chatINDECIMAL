from pathlib import Path

# -------- CONFIG --------
DOCS_PATH = Path("data/documents")
CHUNK_SIZE = 400      # number of words per chunk
CHUNK_OVERLAP = 50    # overlap to preserve context


def load_markdown_files(folder_path):
    """
    Reads all .md files from the given folder
    """
    documents = []

    for file_path in folder_path.glob("*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text, chunk_size, overlap):
    """
    Splits text into overlapping word chunks
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)

        chunks.append(chunk)
        start = end - overlap

    return chunks


def prepare_chunks():
    """
    Full ingestion pipeline:
    documents -> chunks with metadata
    """
    documents = load_markdown_files(DOCS_PATH)
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(
            doc["text"],
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": idx,
                "content": chunk
            })

    return all_chunks


if __name__ == "__main__":
    chunks = prepare_chunks()
    print(f"Total chunks created: {len(chunks)}\n")
    print("Sample chunk:\n")
    print(chunks[0])
