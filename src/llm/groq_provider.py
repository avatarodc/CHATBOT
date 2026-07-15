"""Implementation Groq de LLMProvider (API cloud, compatible OpenAI)."""

import os

import groq

from src.llm.base import LLMProvider, LLMProviderError

MODELE_GROQ = "llama-3.1-8b-instant"
TIMEOUT_SECONDES = 15.0


class GroqProvider(LLMProvider):
    def __init__(self) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMProviderError("GROQ_API_KEY n'est pas definie. Verifiez le fichier .env.")
        self._client = groq.AsyncGroq(api_key=api_key, timeout=TIMEOUT_SECONDES)

    async def generate(self, prompt: str, context: list[str], systeme: str | None = None) -> str:
        messages = self._construire_messages(prompt, context, systeme)

        try:
            reponse = await self._client.chat.completions.create(
                model=MODELE_GROQ,
                messages=messages,
            )
        except groq.AuthenticationError as exc:
            raise LLMProviderError(
                "Cle GROQ_API_KEY invalide ou revoquee. Verifiez votre configuration Groq."
            ) from exc
        except groq.RateLimitError as exc:
            raise LLMProviderError(
                "Quota Groq depasse. Reessayez plus tard ou basculez sur LLM_PROVIDER=ollama."
            ) from exc
        except groq.APITimeoutError as exc:
            raise LLMProviderError(
                f"L'API Groq n'a pas repondu dans le delai imparti ({TIMEOUT_SECONDES}s)."
            ) from exc
        except groq.APIConnectionError as exc:
            raise LLMProviderError("Impossible de contacter l'API Groq (probleme reseau).") from exc
        except groq.GroqError as exc:
            raise LLMProviderError("Erreur lors de l'appel a l'API Groq.") from exc

        return reponse.choices[0].message.content
