import pytest
from unittest.mock import patch

from backend.rag.semantic_search import semantic_search


@patch("backend.rag.semantic_search.VectorStore")
@patch("backend.rag.semantic_search.generate_embedding")
def test_semantic_search_returns_best_match(
    mock_embedding,
    mock_vector_store
):

    mock_embedding.return_value = [1.0, 0.0]

    mock_store = mock_vector_store.return_value

    mock_store.search.return_value = {
        "documents": [[
            "Software installation requires IT approval."
        ]],
        "metadatas": [[
            {"source": "it_policy.txt"}
        ]],
        "distances": [[0.2]]
    }

    results = semantic_search(
        "Do I need permission?",
        top_k=1
    )

    assert len(results) == 1
    assert results[0]["document"] == "it_policy.txt"
    assert results[0]["score"] == pytest.approx(0.8)

    mock_embedding.assert_called_once_with(
        "Do I need permission?"
    )

    mock_store.search.assert_called_once_with(
        query_embedding=[1.0, 0.0],
        top_k=1
    )


def test_semantic_search_empty_question():

    with pytest.raises(ValueError):
        semantic_search("   ")


def test_semantic_search_invalid_top_k():

    with pytest.raises(ValueError):
        semantic_search("Hello", top_k=0)