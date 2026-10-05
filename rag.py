from pathlib import Path
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


# ==========================================
# Настройка
# ==========================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

DOCUMENTS_DIR = Path("documents")


# ==========================================
# Загрузка документов
# ==========================================

def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append({
            "filename": file_path.name,
            "text": text
        })

    return documents


# ==========================================
# Chunking
# ==========================================

def split_into_chunks(text, chunk_size=200):
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
# Embeddings
# ==========================================

def create_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# ==========================================
# Similarity
# ==========================================

def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)

    return np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )


# ==========================================
# Создаём vector store
# ==========================================

documents = load_documents()

chunks = []

for document in documents:

    document_chunks = split_into_chunks(
        document["text"]
    )

    for chunk in document_chunks:

        embedding = create_embedding(chunk)

        chunks.append({
            "filename": document["filename"],
            "text": chunk,
            "embedding": embedding
        })


print(
    f"Загружено chunks: {len(chunks)}"
)


# ==========================================
# Вопрос пользователя
# ==========================================

question = input(
    "\nВаш вопрос: "
)


# ==========================================
# Embedding вопроса
# ==========================================

question_embedding = create_embedding(
    question
)


# ==========================================
# Semantic Search
# ==========================================

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


results.sort(
    key=lambda x: x["similarity"],
    reverse=True
)


# Берём лучшие 3 chunks
top_chunks = results[:3]


# ==========================================
# Формируем Context
# ==========================================

context_parts = []

for i, result in enumerate(
    top_chunks,
    start=1
):

    context_parts.append(
        f"[Source {i}]\n"
        f"{result['text']}"
    )


context = "\n\n".join(
    context_parts
)


# ==========================================
# Prompt для LLM
# ==========================================

prompt = f"""
Ты — помощник, который отвечает
на вопросы только на основании
предоставленного контекста.

Если ответа нет в контексте,
скажи, что информации недостаточно.

Не придумывай факты.

Контекст:

{context}


Вопрос:

{question}
"""


# ==========================================
# Генерация ответа
# ==========================================

response = client.responses.create(
    model="gpt-5.6",
    input=prompt
)


answer = response.output_text


# ==========================================
# Результат
# ==========================================

print("\n" + "=" * 60)

print("ОТВЕТ:")

print(answer)

print("\n" + "=" * 60)

print("ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ:")

for result in top_chunks:

    print(
        f"\n[{result['filename']}] "
        f"(similarity={result['similarity']:.4f})"
    )

    print(result["text"])