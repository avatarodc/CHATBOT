"""Test d'integration : necessite SUPABASE_DB_URL valide (voir .env)."""

import asyncio
import os
import uuid

import pytest

from src.db.connection import close_pool
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
        await apply_schema()

        nom_fichier = f"test_{uuid.uuid4().hex}.pdf"
        document_id = await insert_document(nom_fichier)

        embedding = [0.01] * EMBEDDING_DIM
        chunk_id = await insert_chunk(document_id, "contenu de test factice", embedding)

        resultats = await search_similar_chunks(embedding, top_k=5)

        assert any(r["id"] == chunk_id for r in resultats)

        await close_pool()

    asyncio.run(scenario())
