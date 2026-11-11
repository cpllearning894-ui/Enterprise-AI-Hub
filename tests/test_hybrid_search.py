
from backend.rag.hybrid_search import (
    reciprocal_rank_fusion,
    hybrid_search
)


def test_rrf_combines_matching_results():

    keyword_results = [
        {
            "document": "security_policy.txt",
            "chunk": "MFA is required.",
            "score": 3
        }
    ]

    semantic_results = [
        {
            "document": "security_policy.txt",
            "chunk": "MFA is required.",
            "score": 0.90
        }
    ]

    results = reciprocal_rank_fusion(
        keyword_results,
        semantic_results
    )

    assert len(results) == 1

    assert results[0]["keyword_rank"] == 1
    assert results[0]["semantic_rank"] == 1

    assert round(results[0]["rrf_score"], 6) == 0.032787


def test_rrf_keeps_results_from_both_searches():

    keyword_results = [
        {
            "document": "company_policy.txt",
            "chunk": "Annual leave policy.",
            "score": 2
        }
    ]

    semantic_results = [
        {
            "document": "security_policy.txt",
            "chunk": "MFA is required.",
            "score": 0.90
        }
    ]

    results = reciprocal_rank_fusion(
        keyword_results,
        semantic_results
    )

    assert len(results) == 2

    assert results[0]["keyword_rank"] == 1
    assert results[0]["semantic_rank"] is None

    assert results[1]["keyword_rank"] is None
    assert results[1]["semantic_rank"] == 1


def test_hybrid_search_rejects_empty_question():

    try:
        hybrid_search("   ")
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "Question cannot be empty"


def test_hybrid_search_rejects_invalid_top_k():

    try:
        hybrid_search(
            "Is MFA required?",
            top_k=0
        )
        assert False, "Expected ValueError"
    except ValueError as error:
        assert str(error) == "top_k must be greater than zero"


def test_rrf_preserves_semantic_similarity_score():

    keyword_results = [
        {
            "document": "security_policy.txt",
            "chunk": "MFA is required.",
            "score": 3
        }
    ]

    semantic_results = [
        {
            "document": "security_policy.txt",
            "chunk": "MFA is required.",
            "score": 0.90
        }
    ]

    results = reciprocal_rank_fusion(
        keyword_results,
        semantic_results
    )

    assert len(results) == 1
    assert results[0]["semantic_score"] == 0.90


def test_hybrid_search_rejects_keyword_only_results(
    monkeypatch
):

    import backend.rag.hybrid_search as hybrid_module

    monkeypatch.setattr(
        hybrid_module,
        "keyword_search",
        lambda question, top_k: [
            {
                "document": "company_policy.txt",
                "chunk": "ENTERPRISE AI HUB - COMPANY POLICY",
                "score": 1,
                "chunk_index": 0
            }
        ]
    )

    monkeypatch.setattr(
        hybrid_module,
        "semantic_search",
        lambda question, top_k, similarity_threshold: []
    )

    results = hybrid_module.hybrid_search(
        "What is the company's office address?"
    )

    assert results == []


def test_rrf_preserves_keyword_score():

    keyword_results = [
        {
            "document": "company_policy.txt",
            "chunk": "Employees receive annual leave.",
            "score": 3
        }
    ]

    semantic_results = []

    results = reciprocal_rank_fusion(
        keyword_results,
        semantic_results
    )

    assert len(results) == 1
    assert results[0]["keyword_score"] == 3
    assert results[0]["semantic_score"] is None