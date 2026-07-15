"""Interface commune (Strategy) pour les fournisseurs LLM interchangeables."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError
