
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

    monkeypatch.setattr(
        "backend.rag.rag_evaluator.semantic_search",
        lambda question, top_k: fake_results.pop(0)
    )

    # Run evaluation
    report = evaluate_retrieval(top_k=3)

    # Verify metrics
    assert report["total_questions"] == 3
    assert report["successful_hits"] == 2

    assert report["hit_rate"] == round(2 / 3, 4)

    assert report["mrr"] == round(
        (1 + 0.5 + 0) / 3,
        4
    )

    # Verify individual rankings
    assert report["results"][0]["rank"] == 1
    assert report["results"][1]["rank"] == 2
    assert report["results"][2]["rank"] is None