
from backend.config import OLLAMA_MODEL, OLLAMA_TIMEOUT


def test_ollama_model_configuration():
    assert isinstance(OLLAMA_MODEL, str)
    assert OLLAMA_MODEL.strip() != ""


def test_ollama_timeout_configuration():
    assert isinstance(OLLAMA_TIMEOUT, float)
    assert OLLAMA_TIMEOUT > 0