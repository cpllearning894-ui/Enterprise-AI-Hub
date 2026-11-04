
from backend.rag.document_loader import load_document


def split_document(
    text: str,
    chunk_size: int = 200
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if not text.strip():
        return []

    chunks = []

    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Keep short paragraphs together
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue

        # Split paragraphs that are too large
        for start in range(0, len(paragraph), chunk_size):

            chunk = paragraph[start:start + chunk_size]

            if chunk.strip():
                chunks.append(chunk.strip())

    return chunks


if __name__ == "__main__":

    document = load_document("company_policy.txt")

    chunks = split_document(
        document,
        chunk_size=200
    )

    print("Total Chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):

        print(f"\n--- Chunk {index} ---")
        print(chunk)