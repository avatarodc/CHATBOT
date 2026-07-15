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


async def insert_chunk(
    document_id: int,
    contenu: str,
    embedding: list[float],
    numero_page: int | None = None,
    position: int | None = None,
) -> int:
    """Insere un chunk (contenu + embedding + metadonnees) et retourne son id."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO chunks (document_id, contenu, embedding, numero_page, position)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            document_id,
            contenu,
            embedding,
            numero_page,
            position,
        )
    return row["id"]


async def search_similar_chunks(embedding_question: list[float], top_k: int = 5) -> list[dict]:
    """Retourne les top_k chunks les plus proches par similarite cosinus (operateur <=>)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.document_id, d.nom_fichier, c.numero_page, c.position,
                   c.contenu, c.embedding <=> $1 AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> $1
            LIMIT $2
            """,
            embedding_question,
            top_k,
        )
    return [dict(row) for row in rows]
