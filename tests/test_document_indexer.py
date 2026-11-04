
from unittest.mock import patch, MagicMock

from backend.rag.document_indexer import (
    index_documents,
    calculate_content_hash
)


def test_calculate_content_hash():
    content = "Hello Enterprise AI Hub"

    first_hash = calculate_content_hash(content)
    second_hash = calculate_content_hash(content)

    assert first_hash == second_hash


def test_unchanged_document_is_skipped():

    content = "Employees need IT approval."
    content_hash = calculate_content_hash(content)

    fake_documents = [
        {
            "filename": "it_policy.txt",
            "content": content
        }
    ]

    mock_store = MagicMock()

    mock_store.get_documents_by_source.return_value = {
        "ids": ["it_policy.txt_chunk_0"],
        "metadatas": [
            {"content_hash": content_hash}
        ]
    }

    with patch(
        "backend.rag.document_indexer.load_knowledge_base",
        return_value=fake_documents
    ), patch(
        "backend.rag.document_indexer.VectorStore",
        return_value=mock_store
    ), patch(
        "backend.rag.document_indexer.generate_embedding"
    ) as mock_embedding:

        result = index_documents()

    assert result == 1

    mock_embedding.assert_not_called()
    mock_store.delete_by_source.assert_not_called()


def test_updated_document_is_reindexed():

    old_content = "Employees need IT approval."
    new_content = "Employees need manager and IT approval."

    old_hash = calculate_content_hash(old_content)

    fake_documents = [
        {
            "filename": "it_policy.txt",
            "content": new_content
        }
    ]

    mock_store = MagicMock()

    mock_store.get_documents_by_source.return_value = {
        "ids": ["it_policy.txt_chunk_0"],
        "metadatas": [
            {"content_hash": old_hash}
        ]
    }

    with patch(
        "backend.rag.document_indexer.load_knowledge_base",
        return_value=fake_documents
    ), patch(
        "backend.rag.document_indexer.VectorStore",
        return_value=mock_store
    ), patch(
        "backend.rag.document_indexer.generate_embedding",
        return_value=[0.1, 0.2, 0.3]
    ) as mock_embedding:

        result = index_documents()

    assert result > 0

    mock_embedding.assert_called()

    mock_store.delete_by_source.assert_called_once_with(
        "it_policy.txt"
    )

    mock_store.add_documents.assert_called_once()



def test_deleted_document_is_removed():

    mock_store = MagicMock()

    # Simulate an empty knowledge base
    fake_documents = []

    # Simulate an old document still stored in ChromaDB
    mock_store.get_all_metadata.return_value = {
        "ids": ["old_policy_chunk_0"],
        "metadatas": [
            {"source": "old_policy.txt"}
        ]
    }

    with patch(
        "backend.rag.document_indexer.load_knowledge_base",
        return_value=fake_documents
    ), patch(
        "backend.rag.document_indexer.VectorStore",
        return_value=mock_store
    ):

        result = index_documents()

    assert result == 0

    mock_store.delete_by_source.assert_called_once_with(
        "old_policy.txt"
    )



def test_embedding_failure_preserves_old_records():

    old_content = "Old company policy."
    new_content = "Updated company policy."

    old_hash = calculate_content_hash(old_content)

    fake_documents = [
        {
            "filename": "company_policy.txt",
            "content": new_content
        }
    ]

    mock_store = MagicMock()

    mock_store.get_documents_by_source.return_value = {
        "ids": ["company_policy.txt_chunk_0"],
        "metadatas": [
            {"content_hash": old_hash}
        ]
    }

    with patch(
        "backend.rag.document_indexer.load_knowledge_base",
        return_value=fake_documents
    ), patch(
        "backend.rag.document_indexer.VectorStore",
        return_value=mock_store
    ), patch(
        "backend.rag.document_indexer.generate_embedding",
        side_effect=RuntimeError("Ollama is unavailable")
    ):

        import pytest

        with pytest.raises(
            RuntimeError,
            match="Ollama is unavailable"
        ):
            index_documents()

    # Old records must not be deleted
    mock_store.delete_by_source.assert_not_called()

    # New records must not be saved
    mock_store.add_documents.assert_not_called()