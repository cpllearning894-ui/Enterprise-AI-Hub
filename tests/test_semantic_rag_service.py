
import pytest
from unittest.mock import patch

from backend.rag.semantic_rag_service import (
    answer_semantic_question
)


@patch(
    "backend.rag.semantic_rag_service.OllamaProvider"
)
@patch(
    "backend.rag.semantic_rag_service.hybrid_search"
)
def test_semantic_rag_success(
    mock_search,
    mock_provider
):

    mock_search.return_value = [
        {
            "document": "it_policy.txt",
            "chunk": "IT approval is required.",
            "rrf_score": 0.032,
            "keyword_rank": 1,
            "semantic_rank": 1
        },
        {
            "document": "company_policy.txt",
            "chunk": "Employees must follow IT policies.",
            "rrf_score": 0.016,
            "keyword_rank": 2,
            "semantic_rank": None
        }
    ]

    mock_provider.return_value.generate_response.return_value = (
        "Yes, IT approval is required."
    )

    result = answer_semantic_question(
        "Do I need permission?"
    )

    # Check the generated answer
    assert result["answer"] == (
        "Yes, IT approval is required."
    )

    # Check that both sources are returned
    assert len(result["sources"]) == 2

    assert result["sources"][0]["document"] == (
        "it_policy.txt"
    )

    assert result["sources"][1]["document"] == (
        "company_policy.txt"
    )

    # Check that Hybrid Search was called
    mock_search.assert_called_once_with(
        "Do I need permission?",
        top_k=3
    )

    # Check evaluation results
    assert "evaluation" in result

    evaluation = result["evaluation"]

    assert "faithfulness" in evaluation
    assert "answer_relevance" in evaluation
    assert "context_relevance" in evaluation

    assert evaluation["evaluation_method"] == (
        "keyword_overlap_baseline"
    )

    assert 0.0 <= evaluation["faithfulness"] <= 1.0
    assert 0.0 <= evaluation["answer_relevance"] <= 1.0
    assert 0.0 <= evaluation["context_relevance"] <= 1.0


@patch(
    "backend.rag.semantic_rag_service.hybrid_search"
)
def test_semantic_rag_no_results(mock_search):

    mock_search.return_value = []

    result = answer_semantic_question(
        "Unknown company question"
    )

    assert result["sources"] == []

    assert "evaluation" in result

    assert result["evaluation"]["faithfulness"] == 0.0


def test_semantic_rag_empty_question():

    with pytest.raises(ValueError):
        answer_semantic_question("   ")