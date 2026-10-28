
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


## Local AI Integration with Ollama

Enterprise AI Hub supports local AI inference using Ollama and the Qwen2.5 1.5B language model.

### Requirements

* Python 3.12
* Ollama installed
* Qwen2.5 1.5B model downloaded

### Download the AI Model

```powershell
ollama pull qwen2.5:1.5b
```

### Configure the AI Gateway

Create a `.env` file using `.env.example` as a template.

```dotenv
AI_GATEWAY_MODE=simulation
APP_ENV=development
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_TIMEOUT=120
```

### Run the Backend

```powershell
python -m uvicorn backend.main:app --reload
```

Open Swagger UI:

http://127.0.0.1:8000/docs

### Test the Real AI Provider

Use `POST /api/v1/gateway/chat` with:

```json
{
  "provider": "ollama",
  "message": "Explain artificial intelligence in one sentence."
}
```

The gateway sends the message to the local Ollama model and returns the generated response.

### Run Automated Tests

```powershell
python -m pytest -v
```

The test suite covers gateway validation, simulated providers, Ollama integration, connection failures, timeouts, and configuration checks.

### Security Notes

* Do not commit `.env` files containing secrets.
* Do not hardcode API keys or credentials.
* Review and update Python dependencies regularly.
* Use dependency security scanning before deployment.
