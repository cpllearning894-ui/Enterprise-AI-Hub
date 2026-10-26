
from backend.gateway.providers.base import AIProvider


class GeminiProvider(AIProvider):

    def generate_response(self, message: str) -> str:
        return "Hello from Gemini simulation!"
  