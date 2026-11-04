
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from backend.rag.rag_service import answer_question
from backend.gateway.providers.exceptions import AIProviderError
from backend.rag.semantic_rag_service import answer_semantic_question
from backend.rag.embedding_service import EmbeddingServiceError


router = APIRouter(
    prefix="/api/v1/rag",
    tags=["RAG"]
)


class RAGRequest(BaseModel):
    question: str = Field(max_length=1000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value


@router.post("/ask")
def ask_rag(request: RAGRequest):

    try:
        result = answer_question(request.question)
        return result

    except AIProviderError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.message
        ) from error

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="The knowledge base document is unavailable."
        ) from None



@router.post("/semantic-ask")
def ask_semantic_rag(request: RAGRequest):

    try:
        result = answer_semantic_question(
            request.question
        )

        return result

    except AIProviderError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.message
        ) from error

    except EmbeddingServiceError:
        raise HTTPException(
            status_code=503,
            detail="Embedding service unavailable."
        ) from None

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base unavailable."
        ) from None