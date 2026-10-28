
import os
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


AI_GATEWAY_MODE = os.getenv(
    "AI_GATEWAY_MODE",
    "simulation"
)

APP_ENV = os.getenv(
    "APP_ENV",
    "development"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen2.5:1.5b"
)

OLLAMA_TIMEOUT = float(
    os.getenv("OLLAMA_TIMEOUT", "120")
)


if __name__ == "__main__":
    print("AI Gateway Mode:", AI_GATEWAY_MODE)
    print("Application Environment:", APP_ENV)
    print("Ollama Model:", OLLAMA_MODEL)
    print("Ollama Timeout:", OLLAMA_TIMEOUT)