
from backend.rag.embedding_service import generate_embedding
from backend.rag.vector_store import VectorStore


DEFAULT_SIMILARITY_THRESHOLD = 0.70


def semantic_search(
    question: str,
    top_k: int = 2,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> list[dict]:

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError(
            "similarity_threshold must be between 0 and 1"
        )

    # Step 1: Convert the question into an embedding
    question_vector = generate_embedding(question)

    # Step 2: Connect to the persistent vector database
    store = VectorStore()

    # Step 3: Search ChromaDB
    results = store.search(
        query_embedding=question_vector,
        top_k=top_k
    )

    scored_chunks = []

    # Step 4: Extract search results
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Step 5: Calculate similarity and filter weak results
    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        similarity_score = 1 - distance

        if similarity_score < similarity_threshold:
            continue

        scored_chunks.append({
            "document": metadata["source"],
            "chunk": document,
            "score": similarity_score
        })

    # Step 6: Return relevant chunks only
    return scored_chunks


if __name__ == "__main__":

    question = (
        "Do employees need to use multi-factor authentication?"
    )

    results = semantic_search(
        question,
        top_k=3,
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD
    )

    print("\nQUESTION:", question)

    for index, result in enumerate(results, start=1):

        print(f"\n--- Result {index} ---")
        print("Document:", result["document"])
        print("Similarity:", round(result["score"], 4))
        print("Chunk:", result["chunk"])