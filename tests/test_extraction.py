"""Tests de src.ingestion.extraction."""

from unittest.mock import MagicMock, patch

import pytest
from pypdf.errors import PdfReadError

from src.ingestion.extraction import (
    PDFCorrompuError,
    PDFSansTexteExtractibleError,
    extraire_texte_pdf,
)

PDF_REEL = "data/uploads/ISI_Formations_Test.pdf"


def test_extrait_le_texte_d_un_vrai_pdf():
    pages = extraire_texte_pdf(PDF_REEL)

    assert len(pages) == 2
    assert all(page.texte.strip() for page in pages)
    assert pages[0].numero_page == 1
    assert pages[1].numero_page == 2


def test_pdf_sans_texte_extractible_leve_une_erreur_claire():
    page_sans_texte = MagicMock()
    page_sans_texte.extract_text.return_value = ""
    reader_mock = MagicMock()
    reader_mock.pages = [page_sans_texte, page_sans_texte]

    with patch("src.ingestion.extraction.pypdf.PdfReader", return_value=reader_mock):
        with pytest.raises(PDFSansTexteExtractibleError, match="scanne"):
            extraire_texte_pdf("scanne_sans_texte.pdf")


def test_pdf_corrompu_leve_une_erreur_claire():
    with patch(
        "src.ingestion.extraction.pypdf.PdfReader",
        side_effect=PdfReadError("EOF marker not found"),
    ):
        with pytest.raises(PDFCorrompuError, match="corrompu"):
            extraire_texte_pdf("corrompu.pdf")
