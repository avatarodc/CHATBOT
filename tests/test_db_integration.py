"""Test d'integration : necessite SUPABASE_DB_URL valide (voir .env)."""

import asyncio
import os
import random
import uuid

import pytest

from src.db.connection import close_pool, get_pool
from src.db.migrate import apply_schema
from src.db.repository import insert_chunk, insert_document, search_similar_chunks

EMBEDDING_DIM = 384

requires_supabase = pytest.mark.skipif(
    not os.environ.get("SUPABASE_DB_URL"),
    reason="SUPABASE_DB_URL non configuree dans l'environnement",
)


@requires_supabase
def test_insert_chunk_puis_recherche_similarite():
    async def scenario() -> None:
        document_id = None
        try:
            await apply_schema()

            nom_fichier = f"test_{uuid.uuid4().hex}.pdf"
            document_id = await insert_document(nom_fichier)

            # Vecteur unique a ce run (et non une constante partagee entre executions)
            # pour eviter les ex-aequo de distance avec d'anciennes lignes de test.
            embedding = [random.random() for _ in range(EMBEDDING_DIM)]
            chunk_id = await insert_chunk(
                document_id, "contenu de test factice", embedding, numero_page=1, position=0
            )

            resultats = await search_similar_chunks(embedding, top_k=5)

            assert any(r["id"] == chunk_id for r in resultats)
        finally:
            if document_id is not None:
                pool = await get_pool()
                async with pool.acquire() as conn:
                    await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
            await close_pool()

    asyncio.run(scenario())
