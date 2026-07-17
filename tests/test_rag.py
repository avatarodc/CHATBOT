"""Tests de src.api.rag : orchestration RAG (answer_question)."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.api.rag import (
    MESSAGE_ACCUEIL,
    PROMPT_SYSTEME,
    SEUIL_DISTANCE_MAX,
    _est_une_pure_salutation,
    answer_question,
)

requires_supabase_et_groq = pytest.mark.skipif(
    not os.environ.get("SUPABASE_DB_URL") or not os.environ.get("GROQ_API_KEY", "").startswith("gsk_"),
    reason="SUPABASE_DB_URL et/ou GROQ_API_KEY (valide) non configures",
)


def test_prompt_systeme_charge_depuis_le_fichier():
    contenu_fichier = (Path("src/api/prompt_systeme.md")).read_text(encoding="utf-8")
    assert PROMPT_SYSTEME == contenu_fichier
    assert "Groupe ISI" in PROMPT_SYSTEME


def test_mode_degrade_sans_resultat_pertinent_n_appelle_pas_le_llm():
    resultats_hors_sujet = [
        {"id": 1, "document_id": 1, "nom_fichier": "x.pdf", "numero_page": 1, "position": 0,
         "contenu": "hors sujet", "distance": SEUIL_DISTANCE_MAX + 0.1},
    ]

    with patch("src.api.rag.search_similar_chunks", new=AsyncMock(return_value=resultats_hors_sujet)), \
         patch("src.api.rag.encoder", return_value=[[0.0] * 384]), \
         patch("src.api.rag.get_llm_provider") as get_provider_mock:
        resultat = asyncio.run(answer_question("Quelle est la capitale du Japon ?"))

    get_provider_mock.assert_not_called()
    assert resultat["provider"] is None
    assert resultat["sources"] == []
    assert "n'ai pas cette information" in resultat["reponse"]


@pytest.mark.parametrize(
    "message",
    [
        "Bonjour",
        "bonjour !",
        "Salut",
        "BONSOIR",
        "slt",
        "bjr",
        "Nangadéf",
        "Ça va ?",
        "hello",
        "coucou !!",
    ],
)
def test_pure_salutation_est_detectee(message):
    assert _est_une_pure_salutation(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Bonjour, quelles formations propose l'ISI ?",
        "Quelle est la capitale du Japon ?",
        "Comment ça va ?",
        "Bonjour, je voudrais des informations sur les campus",
        "",
    ],
)
def test_vraie_question_n_est_pas_une_salutation(message):
    assert _est_une_pure_salutation(message) is False


def test_answer_question_normalise_la_casse_avant_l_embedding():
    resultats_hors_sujet = [
        {"id": 1, "document_id": 1, "nom_fichier": "x.pdf", "numero_page": 1, "position": 0,
         "contenu": "hors sujet", "distance": SEUIL_DISTANCE_MAX + 0.1},
    ]

    with patch("src.api.rag.search_similar_chunks", new=AsyncMock(return_value=resultats_hors_sujet)), \
         patch("src.api.rag.encoder", return_value=[[0.0] * 384]) as encoder_mock:
        asyncio.run(answer_question("C'est quoi ISI ?"))

    textes_encodes = encoder_mock.call_args[0][0]
    assert textes_encodes == ["c'est quoi isi ?"]


def test_salutation_repond_par_l_accueil_sans_recherche_ni_llm():
    with patch("src.api.rag.search_similar_chunks", new=AsyncMock()) as recherche_mock, \
         patch("src.api.rag.get_llm_provider") as get_provider_mock:
        resultat = asyncio.run(answer_question("Bonjour"))

    recherche_mock.assert_not_called()
    get_provider_mock.assert_not_called()
    assert resultat["reponse"] == MESSAGE_ACCUEIL
    assert resultat["sources"] == []
    assert resultat["provider"] is None


@requires_supabase_et_groq
def test_answer_question_avec_resultat_pertinent():
    resultat = asyncio.run(answer_question("Quels sont les departements du Groupe ISI ?"))

    assert resultat["reponse"].strip()
    assert resultat["provider"] == "GroqProvider"
    assert resultat["sources"]
    assert all(s["document"] and s["chunk_id"] for s in resultat["sources"])
