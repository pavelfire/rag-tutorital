from pathlib import Path


DOCUMENTS_DIR = Path("documents")


def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "filename": file_path.name,
            "text": text
        })

    return documents


def split_into_chunks(text, chunk_size=100):
    chunks = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

    return chunks


documents = load_documents()

for document in documents:
    chunks = split_into_chunks(document["text"])

    print(f"\nФайл: {document['filename']}")
    print(f"Количество chunks: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(chunk)