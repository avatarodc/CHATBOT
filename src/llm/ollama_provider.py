"""Implementation Ollama de LLMProvider (API locale, http://localhost:11434)."""

import os

import httpx

from src.llm.base import LLMProvider, LLMProviderError

OLLAMA_URL_PAR_DEFAUT = "http://localhost:11434"
TIMEOUT_SECONDES = 180.0


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self._modele = os.environ.get("OLLAMA_MODEL")
        if not self._modele:
            raise LLMProviderError("OLLAMA_MODEL n'est pas definie. Verifiez le fichier .env.")
        self._url_base = os.environ.get("OLLAMA_URL", OLLAMA_URL_PAR_DEFAUT)

    async def generate(self, prompt: str, context: list[str]) -> str:
        messages = self._construire_messages(prompt, context)

        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDES) as client:
                reponse = await client.post(
                    f"{self._url_base}/api/chat",
                    json={"model": self._modele, "messages": messages, "stream": False},
                )
                reponse.raise_for_status()
        except httpx.ConnectError as exc:
            raise LLMProviderError(
                "Ollama n'est pas accessible sur "
                f"{self._url_base}. Lancez `ollama serve` ou demarrez l'application Ollama."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMProviderError(
                f"Ollama n'a pas repondu dans le delai imparti ({TIMEOUT_SECONDES}s)."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Erreur Ollama (modele '{self._modele}') : "
                f"verifiez qu'il est bien pulle (`ollama pull {self._modele}`)."
            ) from exc

        return reponse.json()["message"]["content"]
