
from backend.gateway.providers.openai_provider import OpenAIProvider
from backend.gateway.providers.gemini_provider import GeminiProvider
from backend.gateway.providers.claude_provider import ClaudeProvider


def get_provider(provider_name: str):

    providers = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "claude": ClaudeProvider
    }

    provider_class = providers.get(provider_name)

    if provider_class is None:
        raise ValueError(
            f"Unsupported AI provider: {provider_name}"
        )

    return provider_class()