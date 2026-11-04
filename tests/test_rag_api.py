
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_rag_api_success():

    fake_result = {
        "question": "What is the annual leave policy?",
        "answer": "Employees receive 18 days of paid annual leave.",
        "sources": [
            {
                "document": "company_policy.txt",
                "chunk": "Employees receive 18 days of paid annual leave."
            }
        ]
    }

    with patch(
        "backend.rag.router.answer_question",
        return_value=fake_result
    ):
        response = client.post(
            "/api/v1/rag/ask",
            json={
                "question": "What is the annual leave policy?"
            }
        )

    assert response.status_code == 200
    assert response.json()["answer"] == fake_result["answer"]


def test_rag_api_empty_question():

    response = client.post(
        "/api/v1/rag/ask",
        json={
            "question": "   "
        }
    )

    assert response.status_code == 422


def test_rag_api_missing_question():

    response = client.post(
        "/api/v1/rag/ask",
        json={}
    )

    assert response.status_code == 422


def test_rag_api_long_question():

    response = client.post(
        "/api/v1/rag/ask",
        json={
            "question": "A" * 1001
        }
    )

    assert response.status_code == 422