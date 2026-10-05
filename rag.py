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


# ==========================================
# ChromaDB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_collection(
    name="company_documents"
)


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
# Вопрос
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
# Поиск в ChromaDB
# ==========================================

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)


# ==========================================
# Получаем chunks
# ==========================================

documents = results["documents"][0]

metadatas = results["metadatas"][0]


# ==========================================
# Context
# ==========================================

context_parts = []


for i, document in enumerate(
    documents,
    start=1
):

    context_parts.append(
        f"[Source {i}]\n"
        f"{document}"
    )


context = "\n\n".join(
    context_parts
)


# ==========================================
# Prompt
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
# Gemini
# ==========================================

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=prompt
)


answer = response.text


# ==========================================
# Ответ
# ==========================================

print("\n" + "=" * 60)

print("ОТВЕТ:")

print(answer)

print("\n" + "=" * 60)

print("ИСТОЧНИКИ:")


for i, metadata in enumerate(
    metadatas,
    start=1
):

    print(
        f"[Source {i}] "
        f"{metadata['filename']}"
    )