
from fastapi import FastAPI
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator
from backend.gateway.router import router as gateway_router

app = FastAPI(
    title="Enterprise AI Hub API",
    description="Enterprise AI Integration Platform",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to Enterprise AI Hub",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Enterprise AI Hub Backend",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/api/v1/info")
def system_info():
    return {
        "project": "Enterprise AI Hub",
        "developer_role": "Applied AI Engineer",
        "features": [
            "AI Gateway",
            "RAG Document Assistant",
            "AI Agents",
            "Enterprise Integration"
        ],
        "development_status": "Day 1 - API Foundation"
    }

class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=1000
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be empty")

        return value.strip()


@app.post("/api/v1/chat")
def chat(request: ChatRequest):
    return {
        "user_message": request.message,
        "ai_response": "Hello! I received your message successfully.",
        "status": "success"
    }

app.include_router(gateway_router)