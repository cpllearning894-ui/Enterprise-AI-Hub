
import pytest

from backend.rag.rag_evaluator import evaluate_retrieval


def test_evaluator_calculates_ranking_metrics(monkeypatch):

    # Create a sample evaluation dataset
    dataset = [
        {
            "question": "Question one",
            "expected_source": "it_policy.txt"
        },
        {
            "question": "Question two",
            "expected_source": "security_policy.txt"
        },
        {
            "question": "Question three",
            "expected_source": "company_policy.txt"
        }
    ]

    # Simulate retrieval results
    fake_results = [
        [
            {"document": "it_policy.txt", "chunk": "IT policy", "score": 0.9}
        ],
        [
            {"document": "company_policy.txt", "chunk": "Company policy", "score": 0.9},
            {"document": "security_policy.txt", "chunk": "Security policy", "score": 0.8}
        ],
        [
            {"document": "security_policy.txt", "chunk": "Security policy", "score": 0.9}
        ]
    ]

    monkeypatch.setattr(
        "backend.rag.rag_evaluator.load_evaluation_dataset",
        lambda: dataset
    )

    # Accept the similarity_threshold parameter
    monkeypatch.setattr(
        "backend.rag.rag_evaluator.semantic_search",
        lambda question, top_k, similarity_threshold: fake_results.pop(0)
    )

    # Run evaluation
    report = evaluate_retrieval(top_k=3)

    # Verify general metrics
    assert report["total_questions"] == 3
    assert report["successful_hits"] == 2

    assert report["hit_rate"] == round(2 / 3, 4)

    assert report["mrr"] == round(
        (1 + 0.5 + 0) / 3,
        4
    )

    # Verify the default threshold
    assert report["similarity_threshold"] == 0.70

    # Verify individual rankings
    assert report["results"][0]["rank"] == 1
    assert report["results"][1]["rank"] == 2
    assert report["results"][2]["rank"] is None


def test_evaluator_passes_selected_threshold_to_search(monkeypatch):

    dataset = [
        {
            "question": "Question one",
            "expected_source": "it_policy.txt"
        }
    ]

    received_thresholds = []

    def fake_semantic_search(
        question,
        top_k,
        similarity_threshold
    ):
        received_thresholds.append(similarity_threshold)

        return [
            {
                "document": "it_policy.txt",
                "chunk": "IT policy",
                "score": 0.9
            }
        ]

    monkeypatch.setattr(
        "backend.rag.rag_evaluator.load_evaluation_dataset",
        lambda: dataset
    )

    monkeypatch.setattr(
        "backend.rag.rag_evaluator.semantic_search",
        fake_semantic_search
    )

    # Run evaluation with threshold 0.60
    report = evaluate_retrieval(
        top_k=3,
        similarity_threshold=0.60
    )

    assert report["similarity_threshold"] == 0.60
    assert received_thresholds == [0.60]
    assert report["hit_rate"] == 1.0
    assert report["mrr"] == 1.0


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_evaluator_rejects_invalid_threshold(threshold):

    with pytest.raises(
        ValueError,
        match="similarity_threshold must be between 0 and 1"
    ):
        evaluate_retrieval(similarity_threshold=threshold)


def test_evaluator_rejects_zero_top_k():

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero"
    ):
        evaluate_retrieval(top_k=0)