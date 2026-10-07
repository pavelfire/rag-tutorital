import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from chromadb.utils.embedding_functions import register_embedding_function
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger("rag")

CHROMA_PATH = os.getenv("CHROMA_PATH", "/app/data/chroma")
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", "documents"))
COLLECTION_NAME = "company_documents"
CHUNK_SIZE = 200
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


@register_embedding_function
class ExternalEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embeddings are produced by Gemini and stored explicitly."""

    def __call__(self, input: Documents) -> Embeddings:
        raise RuntimeError("Pass embeddings explicitly")

    @staticmethod
    def name() -> str:
        return "gemini_external"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "ExternalEmbeddingFunction":
        return ExternalEmbeddingFunction()


_gemini: genai.Client | None = None


def gemini_client() -> genai.Client:
    global _gemini
    if _gemini is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set")
        _gemini = genai.Client(api_key=api_key)
    return _gemini


def create_embedding(text: str) -> list[float]:
    response = gemini_client().models.embed_content(
        model=EMBED_MODEL,
        contents=text,
    )
    return response.embeddings[0].values


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    return [
        text[start : start + chunk_size]
        for start in range(0, len(text), chunk_size)
        if text[start : start + chunk_size].strip()
    ]


def get_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ExternalEmbeddingFunction(),
    )


def ensure_index() -> int:
    collection = get_collection()
    if collection.count() > 0:
        logger.info("Chroma already has %s chunks", collection.count())
        return collection.count()

    if not DOCUMENTS_DIR.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {DOCUMENTS_DIR}")

    chunk_id = 0
    for file_path in sorted(DOCUMENTS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        for chunk in split_into_chunks(text):
            logger.info("Indexing chunk %s from %s", chunk_id, file_path.name)
            collection.upsert(
                ids=[str(chunk_id)],
                embeddings=[create_embedding(chunk)],
                documents=[chunk],
                metadatas=[{"filename": file_path.name}],
            )
            chunk_id += 1

    logger.info("Indexed %s chunks", chunk_id)
    return chunk_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    app.state.index_error = None
    try:
        app.state.chunks = ensure_index()
    except Exception as exc:
        app.state.chunks = 0
        app.state.index_error = str(exc)
        logger.exception("Indexing failed")
    yield


app = FastAPI(title="RAG API", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


@app.get("/")
def health():
    return {
        "status": "ok",
        "chunks": getattr(app.state, "chunks", 0),
        "index_error": getattr(app.state, "index_error", None),
    }


@app.post("/ask")
def ask(body: AskRequest):
    if getattr(app.state, "index_error", None):
        raise HTTPException(status_code=503, detail=app.state.index_error)

    collection = get_collection()
    count = collection.count()
    if count == 0:
        raise HTTPException(status_code=503, detail="Collection is empty")

    question_embedding = create_embedding(body.question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(3, count),
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    if not documents:
        return {"answer": "Информации недостаточно.", "sources": []}

    context = "\n\n".join(
        f"[Source {index}]\n{document}"
        for index, document in enumerate(documents, start=1)
    )
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

{body.question}
"""
    response = gemini_client().models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    sources = []
    for metadata in metadatas:
        filename = (metadata or {}).get("filename")
        if filename and filename not in sources:
            sources.append(filename)

    return {"answer": response.text, "sources": sources}
