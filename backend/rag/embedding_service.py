
import httpx


OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


class EmbeddingServiceError(Exception):
    pass


def generate_embedding(text: str) -> list[float]:

    if not text.strip():
        raise ValueError("Text cannot be empty")

    try:
        response = httpx.post(
            OLLAMA_EMBED_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": text
            },
            timeout=60
        )

        response.raise_for_status()

        data = response.json()

        embedding = data["embeddings"][0]

        return embedding

    except (
        httpx.RequestError,
        httpx.HTTPStatusError,
        KeyError,
        IndexError,
        TypeError,
        ValueError
    ) as error:

        raise EmbeddingServiceError(
            "Failed to generate embedding"
        ) from error


if __name__ == "__main__":

    sentence = (
        "Employees need IT approval to install software."
    )

    vector = generate_embedding(sentence)

    print("Embedding generated successfully!")
    print("Vector dimensions:", len(vector))
    print("First 5 values:", vector[:5])