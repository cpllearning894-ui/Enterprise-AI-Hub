
from backend.rag.document_loader import (
    KNOWLEDGE_BASE_DIR
)


def load_knowledge_base() -> list[dict]:

    documents = []

    # Find all TXT files in our knowledge base
    files = sorted(
        KNOWLEDGE_BASE_DIR.glob("*.txt")
    )

    for file_path in files:

        content = file_path.read_text(
            encoding="utf-8"
        )

        if not content.strip():
            continue

        documents.append({
            "filename": file_path.name,
            "content": content
        })

    return documents


if __name__ == "__main__":

    documents = load_knowledge_base()

    print("Total Documents:", len(documents))

    for document in documents:

        print(
            "Loaded:",
            document["filename"]
        )