"""Tests de src.api.rag : orchestration RAG (answer_question)."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.api.rag import PROMPT_SYSTEME, SEUIL_DISTANCE_MAX, answer_question

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


@requires_supabase_et_groq
def test_answer_question_avec_resultat_pertinent():
    resultat = asyncio.run(answer_question("Quels sont les departements du Groupe ISI ?"))

    assert resultat["reponse"].strip()
    assert resultat["provider"] == "GroqProvider"
    assert resultat["sources"]
    assert all(s["document"] and s["chunk_id"] for s in resultat["sources"])
