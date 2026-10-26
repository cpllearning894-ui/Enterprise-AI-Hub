
from abc import ABC, abstractmethod


class AIProvider(ABC):

    @abstractmethod
    def generate_response(self, message: str) -> str:
        pass