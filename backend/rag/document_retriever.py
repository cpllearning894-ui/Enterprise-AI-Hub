
import re

from backend.rag.document_loader import load_document
from backend.rag.document_chunker import split_document


STOP_WORDS = {
    "a", "an", "the", "is", "are", "do", "does",
    "of", "to", "in", "on", "for", "with", "and",
    "how", "what", "many", "can", "our"
}


def extract_keywords(text: str) -> set[str]:
    words = re.findall(r"\b\w+\b", text.lower())

    return {
        word for word in words
        if word not in STOP_WORDS
    }


def retrieve_chunks(
    question: str,
    chunks: list[str],
    top_k: int = 2
) -> list[str]:

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    question_keywords = extract_keywords(question)

    scored_chunks = []

    for chunk in chunks:
        chunk_keywords = extract_keywords(chunk)

        score = len(
            question_keywords.intersection(chunk_keywords)
        )

        if score > 0:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        chunk
        for score, chunk in scored_chunks[:top_k]
    ]


if __name__ == "__main__":

    document = load_document("company_policy.txt")

    chunks = split_document(
        document,
        chunk_size=200
    )

    question = (
        "How many days of annual leave "
        "do employees receive?"
    )

    results = retrieve_chunks(
        question,
        chunks,
        top_k=2
    )

    print("Question:", question)
    print("Retrieved Chunks:", len(results))

    for index, chunk in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(chunk)