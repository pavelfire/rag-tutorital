
from pathlib import Path

# Находим папку с документами
DOCUMENTS_DIR = Path("documents")

# Загружаем все текстовые файлы
documents = []

for file_path in DOCUMENTS_DIR.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")

    documents.append({
        "filename": file_path.name,
        "text": text
    })

# Проверяем результат
print(f"Загружено документов: {len(documents)}")

for document in documents:
    print(f"\nФайл: {document['filename']}")
    print(document["text"])
