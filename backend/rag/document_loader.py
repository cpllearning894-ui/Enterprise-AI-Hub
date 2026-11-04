
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"


def load_document(filename: str) -> str:

    file_path = KNOWLEDGE_BASE_DIR / filename

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Document not found: {filename}"
        )

    content = file_path.read_text(
        encoding="utf-8"
    )

    return content


if __name__ == "__main__":

    document = load_document(
        "company_policy.txt"
    )

    print(document)