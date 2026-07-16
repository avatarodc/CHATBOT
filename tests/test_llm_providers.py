"""Tests de src.llm : interface commune, gestion d'erreurs, factory."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from groq import AuthenticationError as GroqAuthenticationError

from src.llm.base import LLMProvider
from src.llm.factory import get_llm_provider
from src.llm.groq_provider import GroqProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.base import LLMProviderError


def test_construire_messages_avec_contexte():
    messages = LLMProvider._construire_messages("Quelle est la question ?", ["extrait A", "extrait B"])

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "extrait A" in messages[1]["content"]
    assert "extrait B" in messages[1]["content"]
    assert "Quelle est la question ?" in messages[1]["content"]


def test_construire_messages_sans_contexte():
    messages = LLMProvider._construire_messages("Quelle est la question ?", [])

    assert messages[1]["content"] == "Quelle est la question ?"


def test_groq_provider_erreur_si_cle_manquante(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMProviderError, match="GROQ_API_KEY"):
        GroqProvider()


def test_ollama_provider_erreur_si_modele_manquant(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with pytest.raises(LLMProviderError, match="OLLAMA_MODEL"):
        OllamaProvider()


def test_ollama_provider_erreur_si_serveur_injoignable(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    provider = OllamaProvider()

    async def _raise_connect_error(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    with patch("httpx.AsyncClient.post", side_effect=_raise_connect_error):
        with pytest.raises(LLMProviderError, match="ollama serve"):
            asyncio.run(provider.generate("test", []))


def test_ollama_provider_modele_introuvable_404(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "modele-inexistant")
    provider = OllamaProvider()

    requete = httpx.Request("POST", "http://localhost:11434/api/chat")
    reponse_404 = httpx.Response(404, request=requete, json={"error": "model not found"})

    async def _post(*args, **kwargs):
        return reponse_404

    with patch("httpx.AsyncClient.post", side_effect=_post):
        with pytest.raises(LLMProviderError, match="introuvable"):
            asyncio.run(provider.generate("test", []))


def test_ollama_provider_500_proche_du_timeout_signale_comme_tel(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    monkeypatch.setattr("src.llm.ollama_provider.TIMEOUT_SECONDES", 0.01)
    provider = OllamaProvider()

    requete = httpx.Request("POST", "http://localhost:11434/api/chat")
    reponse_500 = httpx.Response(500, request=requete, json={"error": "internal error"})

    async def _post(*args, **kwargs):
        return reponse_500

    with patch("httpx.AsyncClient.post", side_effect=_post):
        with pytest.raises(LLMProviderError, match="depasse le delai imparti"):
            asyncio.run(provider.generate("test", []))


def test_ollama_provider_500_rapide_signale_comme_erreur_generique(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    provider = OllamaProvider()

    requete = httpx.Request("POST", "http://localhost:11434/api/chat")
    reponse_500 = httpx.Response(500, request=requete, json={"error": "internal error"})

    async def _post(*args, **kwargs):
        return reponse_500

    with patch("httpx.AsyncClient.post", side_effect=_post):
        with pytest.raises(LLMProviderError, match="Erreur interne d'Ollama"):
            asyncio.run(provider.generate("test", []))


def test_groq_provider_erreur_cle_invalide_ne_leak_pas_la_cle(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_valeur_secrete_de_test")
    provider = GroqProvider()

    fausse_reponse = MagicMock(status_code=401)
    erreur = GroqAuthenticationError(
        message="invalid api key", response=fausse_reponse, body=None
    )

    with patch.object(
        provider._client.chat.completions, "create", new=AsyncMock(side_effect=erreur)
    ):
        with pytest.raises(LLMProviderError) as exc_info:
            asyncio.run(provider.generate("test", []))

    assert "gsk_valeur_secrete_de_test" not in str(exc_info.value)


def test_factory_retourne_le_provider_configure(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    assert isinstance(get_llm_provider(), OllamaProvider)


def test_factory_leve_erreur_si_provider_inconnu(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "chatgpt")
    with pytest.raises(LLMProviderError, match="inconnu"):
        get_llm_provider()


requires_groq = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY", "").startswith("gsk_"),
    reason="GROQ_API_KEY non configuree (ou invalide)",
)


@requires_groq
def test_groq_provider_repond_reellement():
    provider = GroqProvider()
    reponse = asyncio.run(provider.generate("Reponds uniquement par le mot OK.", []))
    assert reponse.strip()
