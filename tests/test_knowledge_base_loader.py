
from backend.rag.knowledge_base_loader import (
    load_knowledge_base
)


def test_load_multiple_documents():

    documents = load_knowledge_base()

    filenames = {
        document["filename"]
        for document in documents
    }

    assert "company_policy.txt" in filenames
    assert "it_policy.txt" in filenames
    assert "security_policy.txt" in filenames


def test_documents_have_content():

    documents = load_knowledge_base()

    assert len(documents) >= 3

    for document in documents:
        assert document["content"].strip()


def test_document_structure():

    documents = load_knowledge_base()

    for document in documents:
        assert isinstance(document["filename"], str)
        assert isinstance(document["content"], str)