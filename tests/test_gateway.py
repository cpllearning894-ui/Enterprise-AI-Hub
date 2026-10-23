
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


# Test 1: Valid AI Gateway request
def test_gateway_success():

    response = client.post(
        "/api/v1/gateway/chat",
        json={
            "provider": "openai",
            "message": "Hello AI!"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["provider"] == "openai"
    assert data["status"] == "success"
    assert data["mode"] == "simulation"
    assert "request_id" in data
    assert "timestamp" in data
    assert "response_time_ms" in data


# Test 2: Reject an unsupported provider
def test_invalid_provider():

    response = client.post(
        "/api/v1/gateway/chat",
        json={
            "provider": "unknown_ai",
            "message": "Hello!"
        }
    )

    assert response.status_code == 422


# Test 3: Reject an empty message
def test_empty_message():

    response = client.post(
        "/api/v1/gateway/chat",
        json={
            "provider": "gemini",
            "message": ""
        }
    )

    assert response.status_code == 422


# Test 4: Reject whitespace-only messages
def test_whitespace_message():

    response = client.post(
        "/api/v1/gateway/chat",
        json={
            "provider": "claude",
            "message": "     "
        }
    )

    assert response.status_code == 422