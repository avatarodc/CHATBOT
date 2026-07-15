"""Orchestration RAG : answer_question() assemble recherche semantique et LLM."""

import time
from pathlib import Path

from src.db.repository import search_similar_chunks
from src.embeddings.model import encoder
from src.llm.factory import get_llm_provider

TOP_K = 5

# Calibre empiriquement sur le corpus ISI indexe : questions pertinentes -> distance
# cosinus <= 0.63, questions hors-sujet -> distance >= 0.80. 0.7 separe proprement les deux.
SEUIL_DISTANCE_MAX = 0.7

# Reprend mot pour mot le cas 4.1 du prompt systeme (aucun resultat pertinent).
MESSAGE_MODE_DEGRADE = (
    "Je n'ai pas cette information précise dans ma base. Je te recommande de "
    "vérifier sur le site officiel du Groupe ISI ou de contacter directement l'établissement."
)

# Charge une seule fois, au chargement du module : jamais modifie par l'input utilisateur.
PROMPT_SYSTEME = (Path(__file__).parent / "prompt_systeme.md").read_text(encoding="utf-8")


async def answer_question(question: str) -> dict:
    """Pipeline RAG : embed la question, cherche les chunks pertinents, interroge le LLM.

    Si aucun chunk ne passe le seuil de similarite, applique le mode degrade
    (cas 4.1 du prompt systeme) sans appeler le LLM.
    """
    debut = time.time()

    embedding_question = encoder([question])[0]
    resultats = await search_similar_chunks(embedding_question, top_k=TOP_K)
    resultats_pertinents = [r for r in resultats if r["distance"] <= SEUIL_DISTANCE_MAX]

    if not resultats_pertinents:
        return {
            "reponse": MESSAGE_MODE_DEGRADE,
            "sources": [],
            "provider": None,
            "temps_traitement": time.time() - debut,
        }

    contexte = [r["contenu"] for r in resultats_pertinents]
    provider = get_llm_provider()
    reponse = await provider.generate(question, contexte, systeme=PROMPT_SYSTEME)

    sources = [
        {
            "document": r["nom_fichier"],
            "chunk_id": r["id"],
            "numero_page": r["numero_page"],
            "distance": round(r["distance"], 4),
        }
        for r in resultats_pertinents
    ]

    return {
        "reponse": reponse,
        "sources": sources,
        "provider": type(provider).__name__,
        "temps_traitement": time.time() - debut,
    }
