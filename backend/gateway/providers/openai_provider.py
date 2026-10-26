
from backend.gateway.providers.base import AIProvider


class OpenAIProvider(AIProvider):

    def generate_response(self, message: str) -> str:
        return "Hello from OpenAI simulation!"
  