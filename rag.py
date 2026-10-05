from pathlib import Path
import os

import numpy as np
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


def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# --------------------------------
# 1. Загружаем документы
# --------------------------------

documents = load_documents()


# --------------------------------
# 2. Создаём chunks + embeddings
# --------------------------------

chunks = []

for document in documents:
    document_chunks = split_into_chunks(document["text"])

    for chunk in document_chunks:
        embedding = create_embedding(chunk)

        chunks.append({
            "filename": document["filename"],
            "text": chunk,
            "embedding": embedding
        })


print(f"Всего chunks: {len(chunks)}")


# --------------------------------
# 3. Получаем вопрос пользователя
# --------------------------------

question = input("\nВаш вопрос: ")


# --------------------------------
# 4. Создаём embedding вопроса
# --------------------------------

question_embedding = create_embedding(question)


# --------------------------------
# 5. Сравниваем вопрос со chunks
# --------------------------------

results = []

for chunk in chunks:
    similarity = cosine_similarity(
        question_embedding,
        chunk["embedding"]
    )

    results.append({
        "filename": chunk["filename"],
        "text": chunk["text"],
        "similarity": similarity
    })


# --------------------------------
# 6. Сортируем по similarity
# --------------------------------

results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# --------------------------------
# 7. Показываем лучшие результаты
# --------------------------------

print("\nНаиболее релевантные chunks:\n")

for result in results[:3]:
    print("=" * 60)
    print(f"Similarity: {result['similarity']:.4f}")
    print(f"File: {result['filename']}")
    print(result["text"])