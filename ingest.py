from pathlib import Path
import os

import chromadb
from dotenv import load_dotenv
from google import genai


# ==========================================
# Настройка
# ==========================================

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

DOCUMENTS_DIR = Path("documents")


# ==========================================
# ChromaDB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="company_documents"
)


# ==========================================
# Chunking
# ==========================================

def split_into_chunks(
    text,
    chunk_size=200
):

    chunks = []

    for start in range(
        0,
        len(text),
        chunk_size
    ):

        chunk = text[
            start:start + chunk_size
        ]

        chunks.append(chunk)

    return chunks


# ==========================================
# Embedding
# ==========================================

def create_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# ==========================================
# Index documents
# ==========================================

documents = []

for file_path in DOCUMENTS_DIR.glob("*.txt"):

    text = file_path.read_text(
        encoding="utf-8"
    )

    documents.append({
        "filename": file_path.name,
        "text": text
    })


print(
    f"Найдено документов: {len(documents)}"
)


# ==========================================
# Создаём chunks
# ==========================================

chunk_id = 0


for document in documents:

    chunks = split_into_chunks(
        document["text"]
    )

    for chunk in chunks:

        print(
            f"Индексируем chunk {chunk_id}..."
        )

        embedding = create_embedding(
            chunk
        )

        collection.upsert(
            ids=[str(chunk_id)],

            embeddings=[embedding],

            documents=[chunk],

            metadatas=[{
                "filename": document["filename"]
            }]
        )

        chunk_id += 1


print()
print("=" * 50)
print("Индексация завершена")
print("=" * 50)

print(
    f"Всего chunks: {collection.count()}"
)