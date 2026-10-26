
# Enterprise AI Hub

A Python-based enterprise AI integration platform built with FastAPI.

This project is being developed as a practical Applied AI Engineering portfolio project.

## Project Status

🚧 In active development.

The current AI Gateway uses simulated AI provider responses. Real AI model integration is planned.

## Features

- FastAPI REST API
- Simulated multi-provider AI Gateway
- OpenAI, Gemini, and Claude provider selection
- Request validation using Pydantic
- Unique request ID generation
- UTC request timestamps
- Structured terminal logging
- Gateway processing-time measurement
- Automated API testing using pytest

## Technology Stack

- Python 3.12
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- HTTPX
- Git

## Project Structure

```text
Enterprise-AI-Hub/
├── backend/
│   ├── gateway/
│   │   ├── __init__.py
│   │   └── router.py
│   ├── logger.py
│   └── main.py
├── tests/
│   └── test_gateway.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/cpllearning894-ui/Enterprise-AI-Hub.git
cd Enterprise-AI-Hub
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run the Application

```powershell
python -m uvicorn backend.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## Example Gateway Request

Endpoint:

`POST /api/v1/gateway/chat`

Request:

```json
{
  "provider": "openai",
  "message": "Hello AI!"
}
```

Example response:

```json
{
  "request_id": "example-request-id",
  "timestamp": "2026-10-22T15:51:42+00:00",
  "provider": "openai",
  "user_message": "Hello AI!",
  "ai_response": "Hello from OpenAI simulation!",
  "response_time_ms": 0.38,
  "mode": "simulation",
  "status": "success"
}
```

The request ID, timestamp, and processing time vary between requests.

## Run Automated Tests

```powershell
python -m pytest tests/test_gateway.py -v
```

Current test coverage includes:

- Successful gateway requests
- Invalid AI providers
- Empty messages
- Whitespace-only messages

## Planned Features

- Real AI provider integrations
- Secure API key management
- Authentication and authorization
- Retrieval-Augmented Generation (RAG)
- AI agents and tool integrations
- Database integration
- Docker deployment
- Cloud deployment
- Monitoring and observability

## Purpose

To demonstrate practical backend development, AI integration, API engineering, testing, and enterprise AI platform design.