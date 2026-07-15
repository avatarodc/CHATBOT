"""Test d'integration : process_pdf de bout en bout (necessite SUPABASE_DB_URL)."""

import asyncio
import os

import pytest

from src.db.connection import close_pool, get_pool
from src.embeddings.model import DIMENSION_EMBEDDING
from src.embeddings.pipeline import process_pdf

PDF_REEL = "data/uploads/ISI_Formations_Test.pdf"

requires_supabase = pytest.mark.skipif(
    not os.environ.get("SUPABASE_DB_URL"),
    reason="SUPABASE_DB_URL non configuree dans l'environnement",
)


@requires_supabase
def test_process_pdf_insere_chunks_et_embeddings_en_base():
    async def scenario() -> None:
        resultat = await process_pdf(PDF_REEL)

        assert resultat["nb_chunks"] >= 1
        assert len(resultat["chunk_ids"]) == resultat["nb_chunks"]

        pool = await get_pool()
        async with pool.acquire() as conn:
            dims = await conn.fetch(
                "SELECT vector_dims(embedding) AS dim FROM chunks WHERE document_id = $1",
                resultat["document_id"],
            )

        assert len(dims) == resultat["nb_chunks"]
        assert all(row["dim"] == DIMENSION_EMBEDDING for row in dims)

        await close_pool()

    asyncio.run(scenario())
