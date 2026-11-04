
import pytest

from backend.rag.document_chunker import split_document


def test_split_document():

    text = "First paragraph.\n\nSecond paragraph."

    chunks = split_document(text, chunk_size=100)

    assert chunks == [
        "First paragraph.",
        "Second paragraph."
    ]


def test_empty_document():

    assert split_document("", chunk_size=10) == []


def test_invalid_chunk_size():

    with pytest.raises(ValueError):
        split_document("Hello", chunk_size=0)


def test_chunk_size_limit():

    text = "A" * 25

    chunks = split_document(text, chunk_size=10)

    assert all(len(chunk) <= 10 for chunk in chunks)


def test_annual_leave_not_split():

    text = (
        "Annual Leave:\n"
        "Full-time employees receive 18 days "
        "of paid annual leave per year."
    )

    chunks = split_document(text, chunk_size=200)

    assert len(chunks) == 1
    assert "18 days" in chunks[0]