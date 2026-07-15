"""Extraction du texte d'un PDF, page par page (pypdf)."""

from dataclasses import dataclass
from pathlib import Path

import pypdf
from pypdf.errors import PyPdfError


class PDFCorrompuError(Exception):
    """Le fichier PDF est illisible ou corrompu."""


class PDFSansTexteExtractibleError(Exception):
    """Aucun texte n'a pu etre extrait (PDF scanne/image) - OCR hors perimetre MVP."""


@dataclass
class PageTexte:
    numero_page: int
    texte: str


def extraire_texte_pdf(chemin_pdf: str | Path) -> list[PageTexte]:
    """Extrait le texte de chaque page d'un PDF.

    Leve PDFCorrompuError si le fichier est illisible, et
    PDFSansTexteExtractibleError si aucune page ne contient de texte
    extractible (cas typique d'un PDF scanne/image).
    """
    chemin_pdf = Path(chemin_pdf)

    try:
        reader = pypdf.PdfReader(str(chemin_pdf))
        pages = [
            PageTexte(numero_page=i, texte=page.extract_text() or "")
            for i, page in enumerate(reader.pages, start=1)
        ]
    except (PyPdfError, OSError) as exc:
        raise PDFCorrompuError(
            f"Le fichier PDF '{chemin_pdf.name}' est illisible ou corrompu."
        ) from exc

    texte_total = "".join(p.texte for p in pages).strip()
    if not texte_total:
        raise PDFSansTexteExtractibleError(
            f"Aucun texte extractible dans '{chemin_pdf.name}'. "
            "Le PDF est probablement scanne (image) : l'OCR n'est pas pris en charge (hors perimetre MVP)."
        )

    return pages
