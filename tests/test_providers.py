from backend.gateway.provider_factory import get_provider
from backend.gateway.providers.openai_provider import OpenAIProvider
from backend.gateway.providers.gemini_provider import GeminiProvider
from backend.gateway.providers.claude_provider import ClaudeProvider


def test_openai_provider():

    provider = get_provider("openai")

    assert isinstance(provider, OpenAIProvider)

    response = provider.generate_response("Hello!")

    assert response == "Hello from OpenAI simulation!"


def test_gemini_provider():

    provider = get_provider("gemini")

    assert isinstance(provider, GeminiProvider)

    response = provider.generate_response("Hello!")

    assert response == "Hello from Gemini simulation!"


def test_claude_provider():

    provider = get_provider("claude")

    assert isinstance(provider, ClaudeProvider)

    response = provider.generate_response("Hello!")

    assert response == "Hello from Claude simulation!"


def test_unsupported_provider():

    try:
        get_provider("unknown")

        assert False, "Expected ValueError"

    except ValueError as error:

        assert "Unsupported AI provider" in str(error)