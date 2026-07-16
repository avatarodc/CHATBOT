"""Orchestration RAG : answer_question() assemble recherche semantique et LLM."""

import re
import time
import unicodedata
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

MESSAGE_ACCUEIL = (
    "Bonjour ! Je suis l'assistant du Groupe ISI. Pose-moi une question sur les "
    "formations, les campus, les admissions ou la vie étudiante, je répondrai à "
    "partir des documents indexés."
)

# Vocabulaire de pure politesse : un message compose uniquement de ces mots
# (ponctuation et accents ignores) declenche l'accueil sans recherche ni appel LLM.
MOTS_SALUTATION = {
    "bonjour", "bonsoir", "bonne", "journee", "soiree",
    "salut", "slt", "bjr", "cc", "coucou",
    "hello", "hi", "hey", "yo",
    "nangadef", "nangadeff", "salam", "salamalekum",
    "ca", "va", "cv",
}

# Presence de l'un de ces mots => vraie question, jamais une simple politesse.
MOTS_INTERROGATIFS = {
    "quel", "quels", "quelle", "quelles", "comment", "combien", "pourquoi",
    "quand", "qui", "que", "quoi", "ou",
}

# Longueur max (en mots normalises) d'un message pour etre considere comme
# une pure salutation - une vraie question est presque toujours plus longue.
LONGUEUR_MAX_SALUTATION = 4

# Charge une seule fois, au chargement du module : jamais modifie par l'input utilisateur.
PROMPT_SYSTEME = (Path(__file__).parent / "prompt_systeme.md").read_text(encoding="utf-8")


def _normaliser(texte: str) -> str:
    """Minuscules, sans accents, ponctuation retiree, espaces normalises."""
    texte = unicodedata.normalize("NFKD", texte.strip().lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    texte = re.sub(r"[^\w\s]", " ", texte)
    return re.sub(r"\s+", " ", texte).strip()


def _est_une_pure_salutation(question: str) -> bool:
    """Detecte un message court de pure politesse (bonjour, salut, ca va...)
    sans vraie question, pour repondre par un message d'accueil fixe sans
    passer par la recherche semantique ni le LLM. Reste prudent : des qu'un
    mot interrogatif apparait ou que le message depasse quelques mots, ce
    n'est plus une simple salutation."""
    mots = _normaliser(question).split()
    if not mots or len(mots) > LONGUEUR_MAX_SALUTATION:
        return False
    if any(mot in MOTS_INTERROGATIFS for mot in mots):
        return False
    return all(mot in MOTS_SALUTATION for mot in mots)


async def answer_question(question: str) -> dict:
    """Pipeline RAG : embed la question, cherche les chunks pertinents, interroge le LLM.

    Une pure salutation (bonjour, salut...) recoit un message d'accueil fixe
    sans recherche ni appel LLM. Si aucun chunk ne passe le seuil de
    similarite, applique le mode degrade (cas 4.1 du prompt systeme) sans
    appeler le LLM.
    """
    debut = time.time()

    if _est_une_pure_salutation(question):
        return {
            "reponse": MESSAGE_ACCUEIL,
            "sources": [],
            "provider": None,
            "temps_traitement": time.time() - debut,
        }

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
