
import pytest
from unittest.mock import patch

from backend.rag.semantic_rag_service import (
    answer_semantic_question
)


@patch(
    "backend.rag.semantic_rag_service.OllamaProvider"
)
@patch(
    "backend.rag.semantic_rag_service.semantic_search"
)
def test_semantic_rag_success(
    mock_search,
    mock_provider
):

    mock_search.return_value = [
        {
            "document": "it_policy.txt",
            "chunk": "IT approval is required.",
            "score": 0.85
        }
    ]

    mock_provider.return_value.generate_response.return_value = (
        "Yes, IT approval is required."
    )

    result = answer_semantic_question(
        "Do I need permission?"
    )

    assert result["answer"] == (
        "Yes, IT approval is required."
    )

    assert result["sources"][0]["document"] == (
        "it_policy.txt"
    )


@patch(
    "backend.rag.semantic_rag_service.semantic_search"
)
def test_semantic_rag_no_results(mock_search):

    mock_search.return_value = []

    result = answer_semantic_question(
        "Unknown company question"
    )

    assert result["sources"] == []


def test_semantic_rag_empty_question():

    with pytest.raises(ValueError):
        answer_semantic_question("   ")