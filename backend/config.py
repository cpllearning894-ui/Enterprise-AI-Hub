
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


if __name__ == "__main__":
    print("AI Gateway Mode:", AI_GATEWAY_MODE)
    print("Application Environment:", APP_ENV)