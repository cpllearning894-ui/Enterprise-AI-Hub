
from unittest.mock import patch

import pytest

from backend.rag.rag_service import answer_question


def test_rag_answers_using_context():

    with patch(
        "backend.rag.rag_service.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate_response.return_value = (
            "Employees receive 18 days of paid annual leave."
        )

        result = answer_question(
            "How many days of annual leave do employees receive?"
        )

        assert "18 days" in result["answer"]
        assert result["question"]
        assert len(result["sources"]) > 0

        mock_provider.return_value.generate_response.assert_called_once()


def test_rag_empty_question():

    with pytest.raises(ValueError):
        answer_question("   ")


def test_rag_no_matching_information():

    with patch(
        "backend.rag.rag_service.OllamaProvider"
    ) as mock_provider:

        result = answer_question(
            "What is the spaceship launch code?"
        )

        assert result["sources"] == []
        assert "could not find" in result["answer"].lower()

        mock_provider.assert_not_called()



def test_rag_security_document_source():

    with patch(
        "backend.rag.rag_service.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate_response.return_value = (
            "MFA is required for company accounts."
        )

        result = answer_question(
            "Is MFA required for company accounts?"
        )

        assert any(
            source["document"] == "security_policy.txt"
            and "MFA is required" in source["chunk"]
            for source in result["sources"]
        )


def test_rag_it_document_source():

    with patch(
        "backend.rag.rag_service.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate_response.return_value = (
            "Contact the IT Help Desk."
        )

        result = answer_question(
            "Who should employees contact for password reset?"
        )

        assert any(
            source["document"] == "it_policy.txt"
            and "IT Help Desk" in source["chunk"]
            for source in result["sources"]
        )



def test_rag_filters_weak_matches():

    with patch(
        "backend.rag.rag_service.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate_response.return_value = (
            "Yes, MFA is required."
        )

        result = answer_question(
            "Is MFA required for company accounts?"
        )

        assert len(result["sources"]) == 1

        assert result["sources"][0]["document"] == (
            "security_policy.txt"
        )



def test_rag_software_policy_source():

    with patch(
        "backend.rag.rag_service.OllamaProvider"
    ) as mock_provider:

        mock_provider.return_value.generate_response.return_value = (
            "No, IT approval is required."
        )

        result = answer_question(
            "Are employees allowed to install software without IT approval?"
        )

        assert len(result["sources"]) == 1

        assert result["sources"][0]["document"] == (
            "it_policy.txt"
        )

        assert "IT approval" in (
            result["sources"][0]["chunk"]
        )