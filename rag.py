from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


def split_into_chunks(text, chunk_size=200):
    chunks = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

    return chunks


def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


documents = load_documents()

for document in documents:
    chunks = split_into_chunks(document["text"])

    for i, chunk in enumerate(chunks):
        embedding = create_embedding(chunk)

        print(f"\nChunk {i}:")
        print(chunk)

        print("\nРазмер embedding:")
        print(len(embedding))

        print("\nПервые 5 чисел:")
        print(embedding[:5])