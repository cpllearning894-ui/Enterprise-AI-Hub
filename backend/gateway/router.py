from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from datetime import datetime, timezone
from uuid import uuid4
from backend.logger import logger
import time
from backend.config import AI_GATEWAY_MODE
from backend.gateway.provider_factory import get_provider
from fastapi import HTTPException
from backend.gateway.providers.exceptions import AIProviderError



router = APIRouter(
    prefix="/api/v1/gateway",
    tags=["AI Gateway"]
)


# Define the incoming request
class GatewayRequest(BaseModel):
    provider: Literal["openai", "gemini", "claude", "ollama"]

    message: str = Field(
        min_length=1,
        max_length=1000
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        cleaned = value.strip()

        if not cleaned:
            raise ValueError("Message cannot be empty")

        return cleaned


# Our simulated AI Gateway
@router.post("/chat")
def gateway_chat(request: GatewayRequest):

    start_time = time.perf_counter()

    request_id = str(uuid4())

    logger.info(
        "Gateway request received | request_id=%s | provider=%s",
        request_id,
        request.provider
    )

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()


    provider = get_provider(request.provider)

    try:
        selected_response = provider.generate_response(
            request.message
        )

    except AIProviderError as error:

        logger.error(
            "Gateway request failed | request_id=%s | provider=%s | status_code=%s",
            request_id,
            request.provider,
            error.status_code
        )

        raise HTTPException(

            status_code=error.status_code,
            detail={
                "request_id": request_id,
                "message": error.message,
                "status": "error"
            }
        ) from error

    responses_time_ms = round(
        (time.perf_counter() - start_time) * 1000, 2
    )

    logger.info(
        "Gateway request completed | request_id=%s | status=success",
        request_id
    )

    return {
        "request_id": request_id,
        "timestamp": timestamp,
        "provider": request.provider,
        "user_message": request.message,
        "ai_response": selected_response,
        "response_time_ms": responses_time_ms,
        "mode": "local" if request.provider == "ollama" else AI_GATEWAY_MODE,
        "status": "success"
    }