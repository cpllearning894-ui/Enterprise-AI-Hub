
import pytest

from backend.rag.document_retriever import (
    extract_keywords,
    retrieve_chunks
)


def test_extract_keywords():
    keywords = extract_keywords(
        "What is the annual leave policy?"
    )

    assert "annual" in keywords
    assert "leave" in keywords
    assert "policy" in keywords
    assert "the" not in keywords


def test_retrieve_relevant_chunk():
    chunks = [
        "Employees receive 18 days of annual leave.",
        "IT support handles computer problems.",
        "Remote work requires manager approval."
    ]

    results = retrieve_chunks(
        "What is the annual leave policy?",
        chunks,
        top_k=1
    )

    assert len(results) == 1
    assert "18 days" in results[0]


def test_no_matching_chunks():
    chunks = [
        "Employees work Monday to Friday.",
        "The office opens at 8 AM."
    ]

    results = retrieve_chunks(
        "What is the refund policy?",
        chunks
    )

    assert results == []


def test_invalid_top_k():
    with pytest.raises(ValueError):
        retrieve_chunks(
            "annual leave",
            ["Annual leave is available."],
            top_k=0
        )


def test_top_k_limit():
    chunks = [
        "Annual leave policy one.",
        "Annual leave policy two.",
        "Annual leave policy three."
    ]

    results = retrieve_chunks(
        "annual leave",
        chunks,
        top_k=2
    )

    assert len(results) == 2