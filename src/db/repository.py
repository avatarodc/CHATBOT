"""Acces aux tables documents/chunks. Requetes uniquement parametrees ($1, $2, ...)."""

from src.db.connection import get_pool


async def insert_document(nom_fichier: str) -> int:
    """Insere un document et retourne son id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO documents (nom_fichier) VALUES ($1) RETURNING id",
            nom_fichier,
        )
    return row["id"]


async def insert_chunk(document_id: int, contenu: str, embedding: list[float]) -> int:
    """Insere un chunk (contenu + embedding) rattache a un document et retourne son id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO chunks (document_id, contenu, embedding)
            VALUES ($1, $2, $3)
            RETURNING id
            """,
            document_id,
            contenu,
            embedding,
        )
    return row["id"]


async def search_similar_chunks(embedding_question: list[float], top_k: int = 5) -> list[dict]:
    """Retourne les top_k chunks les plus proches par similarite cosinus (operateur <=>)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, document_id, contenu, embedding <=> $1 AS distance
            FROM chunks
            ORDER BY embedding <=> $1
            LIMIT $2
            """,
            embedding_question,
            top_k,
        )
    return [dict(row) for row in rows]
