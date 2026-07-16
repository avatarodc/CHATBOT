"""Implementation Ollama de LLMProvider (API locale, http://localhost:11434)."""

import os
import time

import httpx

from src.llm.base import LLMProvider, LLMProviderError

OLLAMA_URL_PAR_DEFAUT = "http://localhost:11434"
TIMEOUT_SECONDES = 180.0

# Au-dela de cette fraction du timeout, un statut 500 est traite comme un
# probable abandon interne d'Ollama plutot qu'une autre erreur (modele
# introuvable, corps de reponse invalide, etc.) - constate empiriquement :
# Ollama peut renvoyer 500 (et non une timeout httpx cote client) lorsqu'il
# interrompt lui-meme une generation trop longue pres de la limite configuree.
SEUIL_PROCHE_TIMEOUT = 0.95


class OllamaProvider(LLMProvider):
    def __init__(self) -> None:
        self._modele = os.environ.get("OLLAMA_MODEL")
        if not self._modele:
            raise LLMProviderError("OLLAMA_MODEL n'est pas definie. Verifiez le fichier .env.")
        self._url_base = os.environ.get("OLLAMA_URL", OLLAMA_URL_PAR_DEFAUT)

    async def generate(self, prompt: str, context: list[str], systeme: str | None = None) -> str:
        messages = self._construire_messages(prompt, context, systeme)
        debut = time.monotonic()

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
                f"Ollama n'a pas repondu dans le delai imparti ({TIMEOUT_SECONDES:.0f}s)."
            ) from exc
        except httpx.HTTPStatusError as exc:
            duree = time.monotonic() - debut

            if exc.response.status_code == 404:
                raise LLMProviderError(
                    f"Modele Ollama '{self._modele}' introuvable : "
                    f"verifiez qu'il est bien pulle (`ollama pull {self._modele}`)."
                ) from exc

            if duree >= TIMEOUT_SECONDES * SEUIL_PROCHE_TIMEOUT:
                raise LLMProviderError(
                    f"Ollama a depasse le delai imparti ({TIMEOUT_SECONDES:.0f}s) et a "
                    "interrompu la generation en interne (reponse trop longue a produire "
                    "sur cette machine). Reessayez avec une question plus courte, ou "
                    "augmentez TIMEOUT_SECONDES si le modele est simplement lent."
                ) from exc

            raise LLMProviderError(
                f"Erreur interne d'Ollama (modele '{self._modele}', "
                f"statut HTTP {exc.response.status_code})."
            ) from exc

        return reponse.json()["message"]["content"]
