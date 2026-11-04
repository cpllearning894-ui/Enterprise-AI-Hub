
import pytest
import httpx

from unittest.mock import patch, Mock

from backend.rag.embedding_service import (
    generate_embedding,
    EmbeddingServiceError
)


def test_generate_embedding_success():

    mock_response = Mock()

    mock_response.json.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]]
    }

    with patch(
        "backend.rag.embedding_service.httpx.post",
        return_value=mock_response
    ):

        result = generate_embedding("Hello AI")

        assert result == [0.1, 0.2, 0.3]


def test_generate_embedding_empty_text():

    with pytest.raises(ValueError):

        generate_embedding("   ")



def test_generate_embedding_connection_error():


    with patch(
        "backend.rag.embedding_service.httpx.post",
        side_effect=httpx.ConnectError(
            "Connection failed"
        )
    ):

        with pytest.raises(EmbeddingServiceError):

            generate_embedding("Hello AI")