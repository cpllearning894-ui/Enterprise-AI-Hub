
import json
from pathlib import Path

from backend.rag.semantic_search import semantic_search


# Find the project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Evaluation dataset location
DATASET_PATH = BASE_DIR / "tests" / "rag_evaluation_dataset.json"


def load_evaluation_dataset() -> list[dict]:
    """
    Load evaluation questions and expected source documents.
    """

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset


def evaluate_retrieval(
    top_k: int = 3,
    similarity_threshold: float = 0.70
) -> dict:
    """
    Evaluate semantic retrieval using Hit Rate and MRR.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    if not 0 <= similarity_threshold <= 1:
        raise ValueError(
            "similarity_threshold must be between 0 and 1"
        )

    # Step 1: Load evaluation questions
    dataset = load_evaluation_dataset()

    if not dataset:
        raise ValueError("Evaluation dataset cannot be empty")

    total_questions = len(dataset)

    successful_hits = 0
    reciprocal_rank_sum = 0.0

    evaluation_results = []

    # Step 2: Evaluate every question
    for item in dataset:

        question = item["question"]
        expected_source = item["expected_source"]

        # Retrieve chunks using the selected threshold
        retrieved_chunks = semantic_search(
            question,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Extract unique source filenames
        retrieved_sources = list(dict.fromkeys(
            chunk["document"]
            for chunk in retrieved_chunks
        ))

        # Step 3: Find the expected document's rank
        rank = None

        for index, source in enumerate(retrieved_sources, start=1):

            if source == expected_source:
                rank = index
                break

        # Step 4: Calculate Hit Rate and Reciprocal Rank
        if rank is not None:

            successful_hits += 1
            reciprocal_rank = 1 / rank

        else:

            reciprocal_rank = 0.0

        reciprocal_rank_sum += reciprocal_rank

        evaluation_results.append({
            "question": question,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "rank": rank,
            "hit": rank is not None,
            "reciprocal_rank": reciprocal_rank
        })

    # Step 5: Calculate evaluation metrics
    hit_rate = successful_hits / total_questions

    mean_reciprocal_rank = (
        reciprocal_rank_sum / total_questions
    )

    # Step 6: Return evaluation report
    return {
        "total_questions": total_questions,
        "successful_hits": successful_hits,
        "similarity_threshold": similarity_threshold,
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mean_reciprocal_rank, 4),
        "results": evaluation_results
    }


if __name__ == "__main__":

    thresholds = [0.50, 0.60, 0.70]

    print("\n===== SEMANTIC RETRIEVAL EXPERIMENT =====")

    for threshold in thresholds:

        report = evaluate_retrieval(
            top_k=3,
            similarity_threshold=threshold
        )

        print("\n====================================")
        print("Similarity Threshold:", threshold)
        print("Total Questions:", report["total_questions"])
        print("Successful Hits:", report["successful_hits"])
        print("Hit Rate:", report["hit_rate"])
        print("MRR:", report["mrr"])

        print("\nIndividual Results:")

        for index, result in enumerate(
            report["results"], start=1
        ):

            print(f"\nQuestion {index}:")
            print("Expected:", result["expected_source"])
            print("Retrieved:", result["retrieved_sources"])
            print("Rank:", result["rank"])
            print("Hit:", result["hit"])
            print("Reciprocal Rank:", result["reciprocal_rank"])