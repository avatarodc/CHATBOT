"""Decoupage du texte extrait en chunks (~300-500 mots, chevauchement ~50 mots).

Coupe en priorite sur des frontieres de paragraphe (lignes vides dans le
texte source) ; a defaut, ne coupe jamais au milieu d'une phrase.
"""

import re
from dataclasses import dataclass

from src.ingestion.extraction import PageTexte

TAILLE_MIN_MOTS = 300
TAILLE_MAX_MOTS = 500
CHEVAUCHEMENT_MOTS = 50

_SEPARATEUR_PARAGRAPHE = re.compile(r"\n\s*\n+")
_SEPARATEUR_PHRASE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Unite:
    texte: str
    numero_page: int
    nouveau_paragraphe: bool


@dataclass
class Chunk:
    document_source: str
    numero_page: int
    position: int
    contenu: str


def _nb_mots(texte: str) -> int:
    return len(texte.split())


def _decouper_en_unites(pages: list[PageTexte]) -> list[Unite]:
    """Decoupe chaque page en paragraphes puis en phrases (unite de decoupage minimale)."""
    unites: list[Unite] = []
    for page in pages:
        paragraphes = [p for p in _SEPARATEUR_PARAGRAPHE.split(page.texte) if p.strip()]
        for paragraphe in paragraphes:
            texte_normalise = " ".join(paragraphe.split())
            phrases = [p for p in _SEPARATEUR_PHRASE.split(texte_normalise) if p.strip()]
            for i, phrase in enumerate(phrases):
                unites.append(
                    Unite(texte=phrase, numero_page=page.numero_page, nouveau_paragraphe=(i == 0))
                )
    return unites


def chunker_pages(pages: list[PageTexte], document_source: str) -> list[Chunk]:
    """Assemble les unites (phrases) extraites en chunks avec chevauchement."""
    unites = _decouper_en_unites(pages)

    chunks: list[Chunk] = []
    courant: list[tuple[str, int]] = []

    def mots_dans(liste: list[tuple[str, int]]) -> int:
        return sum(_nb_mots(texte) for texte, _ in liste)

    def cloturer() -> None:
        if not courant:
            return
        chunks.append(
            Chunk(
                document_source=document_source,
                numero_page=courant[0][1],
                position=len(chunks),
                contenu=" ".join(texte for texte, _ in courant),
            )
        )

    def chevauchement(liste: list[tuple[str, int]]) -> list[tuple[str, int]]:
        retenues: list[tuple[str, int]] = []
        total = 0
        for item in reversed(liste):
            retenues.insert(0, item)
            total += _nb_mots(item[0])
            if total >= CHEVAUCHEMENT_MOTS:
                break
        return retenues

    for unite in unites:
        mots_courants = mots_dans(courant)
        mots_unite = _nb_mots(unite.texte)

        depasse_max = bool(courant) and (mots_courants + mots_unite > TAILLE_MAX_MOTS)
        frontiere_paragraphe_atteinte = (
            bool(courant) and mots_courants >= TAILLE_MIN_MOTS and unite.nouveau_paragraphe
        )

        if depasse_max or frontiere_paragraphe_atteinte:
            cloturer()
            courant = chevauchement(courant)

        courant.append((unite.texte, unite.numero_page))

    cloturer()
    return chunks
