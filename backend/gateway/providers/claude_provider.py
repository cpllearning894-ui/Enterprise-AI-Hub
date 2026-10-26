
from backend.gateway.providers.base import AIProvider


class ClaudeProvider(AIProvider):

    def generate_response(self, message: str) -> str:
        return "Hello from Claude simulation!"
  