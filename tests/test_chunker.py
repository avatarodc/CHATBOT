"""Tests de src.ingestion.chunker."""

from src.ingestion.chunker import (
    CHEVAUCHEMENT_MOTS,
    TAILLE_MAX_MOTS,
    _nb_mots,
    chunker_pages,
)
from src.ingestion.extraction import extraire_texte_pdf

PDF_REEL = "data/uploads/ISI_Formations_Test.pdf"


def test_chunks_ont_les_metadonnees_attendues():
    pages = extraire_texte_pdf(PDF_REEL)
    chunks = chunker_pages(pages, "ISI_Formations_Test.pdf")

    assert len(chunks) >= 1
    for i, chunk in enumerate(chunks):
        assert chunk.document_source == "ISI_Formations_Test.pdf"
        assert chunk.position == i
        assert chunk.numero_page >= 1
        assert chunk.contenu.strip()


def test_aucun_chunk_ne_depasse_la_taille_max_de_maniere_deraisonnable():
    pages = extraire_texte_pdf(PDF_REEL)
    chunks = chunker_pages(pages, "ISI_Formations_Test.pdf")

    for chunk in chunks:
        assert _nb_mots(chunk.contenu) <= TAILLE_MAX_MOTS + CHEVAUCHEMENT_MOTS


def test_chevauchement_entre_deux_chunks_consecutifs():
    pages = extraire_texte_pdf(PDF_REEL)
    chunks = chunker_pages(pages, "ISI_Formations_Test.pdf")

    if len(chunks) < 2:
        return

    fin_premier_chunk = chunks[0].contenu[-200:]
    debut_second_chunk = chunks[1].contenu[:200]
    mots_communs = set(fin_premier_chunk.split()) & set(debut_second_chunk.split())
    assert mots_communs
