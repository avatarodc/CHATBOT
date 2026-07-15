"""Implementation Ollama de LLMProvider. Logique a implementer a l'etape LLM."""

from src.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError
