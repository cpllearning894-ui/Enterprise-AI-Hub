
import httpx

from backend.gateway.providers.base import AIProvider
from backend.gateway.providers.exceptions import AIProviderError
from backend.config import OLLAMA_MODEL, OLLAMA_TIMEOUT


class OllamaProvider(AIProvider):

    def generate_response(self, message: str) -> str:

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": message,
            "stream": False
        }

        try:
            response = httpx.post(
                "http://127.0.0.1:11434/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT
            )

            response.raise_for_status()

            data = response.json()

            return data["response"]

        except httpx.TimeoutException:
            raise AIProviderError(
                "The local AI model took too long to respond.",
                status_code=504
            ) from None

        except httpx.ConnectError:
            raise AIProviderError(
                "Cannot connect to Ollama. Check that Ollama is running.",
                status_code=503
            ) from None

        except httpx.HTTPStatusError:
            raise AIProviderError(
                "Ollama returned an unsuccessful response.",
                status_code=502
            ) from None

        except (ValueError, KeyError, TypeError):
            raise AIProviderError(
                "Ollama returned an invalid response.",
                status_code=502
            ) from None

        except httpx.RequestError:
            raise AIProviderError(
                "A network error occurred while contacting Ollama.",
                status_code=503
            ) from None