from backend.rag.embedding_service import generate_embedding
from backend.rag.vector_store import VectorStore


def semantic_search(
    question: str,
    top_k: int = 2
) -> list[dict]:

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    # Step 1: Convert the user's question into an embedding
    question_vector = generate_embedding(question)

    # Step 2: Connect to the persistent vector database
    store = VectorStore()

    # Step 3: Search ChromaDB for similar document chunks
    results = store.search(
        query_embedding=question_vector,
        top_k=top_k
    )

    scored_chunks = []

    # Step 4: Extract the search results
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Step 5: Convert distances into similarity scores
    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        similarity_score = 1 - distance

        scored_chunks.append({
            "document": metadata["source"],
            "chunk": document,
            "score": similarity_score
        })

    # Step 6: Return the best matches
    return scored_chunks


if __name__ == "__main__":

    question = (
        "Do I need permission to add a program "
        "to my work laptop?"
    )

    results = semantic_search(
        question,
        top_k=2
    )

    print("\nQUESTION:", question)

    for index, result in enumerate(results, start=1):

        print(f"\n--- Result {index} ---")
        print("Document:", result["document"])
        print("Similarity:", round(result["score"], 4))
        print("Chunk:", result["chunk"])