"""Pipeline embeddings : embed_chunks (insertion en base) et process_pdf (bout en bout)."""

from pathlib import Path

from src.db.repository import insert_chunk, insert_document
from src.embeddings.model import encoder, normaliser_pour_embedding
from src.ingestion.chunker import Chunk, chunker_pages
from src.ingestion.extraction import extraire_texte_pdf


async def embed_chunks(chunks: list[Chunk], document_id: int) -> list[int]:
    """Calcule les embeddings des chunks et les insere en base. Retourne les ids inseres."""
    if not chunks:
        return []

    # Seul le texte envoye a l'embedding est normalise : chunk.contenu garde
    # sa casse d'origine pour l'affichage/citation et le contexte envoye au LLM.
    textes = [normaliser_pour_embedding(chunk.contenu) for chunk in chunks]
    embeddings = encoder(textes)

    ids = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = await insert_chunk(
            document_id, chunk.contenu, embedding, chunk.numero_page, chunk.position
        )
        ids.append(chunk_id)
    return ids


async def process_pdf(chemin_pdf: str) -> dict:
    """Extraction + chunking + embeddings + insertion, bout en bout.

    L'extraction et le chunking se font avant toute ecriture en base : si le
    PDF est corrompu ou sans texte extractible, l'exception se propage et
    aucune entree (document ni chunk) n'est creee.
    """
    nom_fichier = Path(chemin_pdf).name

    pages = extraire_texte_pdf(chemin_pdf)
    chunks = chunker_pages(pages, nom_fichier)

    if not chunks:
        raise ValueError(f"Aucun chunk genere pour '{nom_fichier}' apres extraction et decoupage.")

    document_id = await insert_document(nom_fichier)
    chunk_ids = await embed_chunks(chunks, document_id)

    return {
        "document_id": document_id,
        "nom_fichier": nom_fichier,
        "nb_chunks": len(chunk_ids),
        "chunk_ids": chunk_ids,
    }
