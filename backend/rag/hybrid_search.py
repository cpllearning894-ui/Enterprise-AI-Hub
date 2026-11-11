
from backend.rag.document_retriever import (
    extract_keywords
)
from backend.rag.knowledge_base_loader import (
    load_knowledge_base
)
from backend.rag.document_chunker import (
    split_document
)
from backend.rag.semantic_search import semantic_search


DEFAULT_RRF_K = 60
DEFAULT_SIMILARITY_THRESHOLD = 0.60


def keyword_search(
    question: str,
    top_k: int = 3
) -> list[dict]:

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    question_keywords = extract_keywords(question)

    documents = load_knowledge_base()

    results = []

    for document in documents:

        filename = document["filename"]

        chunks = split_document(
            document["content"],
            chunk_size=200
        )

        for index, chunk in enumerate(chunks):

            chunk_keywords = extract_keywords(chunk)

            score = len(
                question_keywords.intersection(
                    chunk_keywords
                )
            )

            if score > 0:
                results.append({
                    "document": filename,
                    "chunk": chunk,
                    "score": score,
                    "chunk_index": index
                })

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]



def reciprocal_rank_fusion(
    keyword_results: list[dict],
    semantic_results: list[dict],
    rrf_k: int = DEFAULT_RRF_K
) -> list[dict]:

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    fused_results = {}

    # Process keyword search rankings
    for rank, result in enumerate(
        keyword_results,
        start=1
    ):

        key = (
            result["document"],
            result["chunk"]
        )

        if key not in fused_results:
            fused_results[key] = {
                "document": result["document"],
                "chunk": result["chunk"],
                "rrf_score": 0.0,
                "keyword_rank": None,
                "semantic_rank": None,
                "semantic_score": None,
                "keyword_score": None,
            }

        fused_results[key]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

        fused_results[key]["keyword_rank"] = rank

        fused_results[key]["keyword_score"] = result["score"]

    # Process semantic search rankings
    for rank, result in enumerate(
        semantic_results,
        start=1
    ):

        key = (
            result["document"],
            result["chunk"]
        )

        if key not in fused_results:
            fused_results[key] = {
                "document": result["document"],
                "chunk": result["chunk"],
                "rrf_score": 0.0,
                "keyword_rank": None,
                "semantic_rank": None,
                "semantic_score": None,
                "keyword_score": None,
            }

        fused_results[key]["rrf_score"] += (
            1 / (rrf_k + rank)
        )

        fused_results[key]["semantic_rank"] = rank

        fused_results[key]["semantic_score"] = (
            result["score"]
        )

    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results


def hybrid_search(
    question: str,
    top_k: int = 3,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    rrf_k: int = DEFAULT_RRF_K
) -> list[dict]:

    if not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError(
            "similarity_threshold must be between 0 and 1"
        )

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    # Step 1: Run keyword search
    keyword_results = keyword_search(
        question,
        top_k=top_k
    )

    # Step 2: Run semantic search
    semantic_results = semantic_search(
        question,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )

    # Step 3: Combine both rankings using RRF
    fused_results = reciprocal_rank_fusion(
        keyword_results,
        semantic_results,
        rrf_k=rrf_k
    )

    # Step 4: Keep results supported by semantic search
    relevant_results = [
        result
        for result in fused_results
        if result["semantic_score"] is not None
    ]

    # Step 5: Return the highest-ranked relevant results
    return relevant_results[:top_k]




if __name__ == "__main__":

    question = (
        "Is multi-factor authentication required "
        "for company accounts?"
    )

    results = hybrid_search(
        question,
        top_k=3
    )

    print("\nQUESTION:", question)
    print("HYBRID RESULTS:", len(results))

    for index, result in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Result {index} ---")
        print("Document:", result["document"])
        print("RRF Score:", round(result["rrf_score"], 6))
        print("Keyword Rank:", result["keyword_rank"])
        print("Semantic Rank:", result["semantic_rank"])
        print("Chunk:", result["chunk"])