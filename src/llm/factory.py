"""Factory : selectionne l'implementation LLMProvider selon LLM_PROVIDER (.env)."""

import os

from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMProviderError
from src.llm.groq_provider import GroqProvider
from src.llm.ollama_provider import OllamaProvider

load_dotenv()

_FOURNISSEURS = {
    "groq": GroqProvider,
    "ollama": OllamaProvider,
}


def get_llm_provider() -> LLMProvider:
    """Retourne l'implementation LLMProvider correspondant a LLM_PROVIDER (.env)."""
    nom = os.environ.get("LLM_PROVIDER", "").strip().lower()
    classe = _FOURNISSEURS.get(nom)
    if classe is None:
        raise LLMProviderError(
            f"LLM_PROVIDER='{nom}' inconnu. Valeurs acceptees : {', '.join(_FOURNISSEURS)}."
        )
    return classe()
