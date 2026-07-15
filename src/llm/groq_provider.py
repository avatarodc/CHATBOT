"""Implementation Groq de LLMProvider. Logique a implementer a l'etape LLM."""

from src.llm.base import LLMProvider


class GroqProvider(LLMProvider):
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError
