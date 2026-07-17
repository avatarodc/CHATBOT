"""Test d'integration : process_pdf de bout en bout (necessite SUPABASE_DB_URL)."""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from src.db.connection import close_pool, get_pool
from src.embeddings.model import DIMENSION_EMBEDDING
from src.embeddings.pipeline import embed_chunks, process_pdf
from src.ingestion.chunker import Chunk

PDF_REEL = "data/uploads/ISI_Formations_Test.pdf"

requires_supabase = pytest.mark.skipif(
    not os.environ.get("SUPABASE_DB_URL"),
    reason="SUPABASE_DB_URL non configuree dans l'environnement",
)


def test_embed_chunks_normalise_la_casse_avant_l_embedding():
    chunks = [Chunk(document_source="x.pdf", numero_page=1, position=0, contenu="C'est quoi ISI ?")]

    with patch("src.embeddings.pipeline.encoder", return_value=[[0.0] * DIMENSION_EMBEDDING]) as encoder_mock, \
         patch("src.embeddings.pipeline.insert_chunk", new=AsyncMock(return_value=1)) as insert_mock:
        asyncio.run(embed_chunks(chunks, document_id=1))

    textes_encodes = encoder_mock.call_args[0][0]
    assert textes_encodes == ["c'est quoi isi ?"]

    # Le contenu stocke garde sa casse d'origine (affichage/citation, contexte LLM).
    contenu_insere = insert_mock.call_args[0][1]
    assert contenu_insere == "C'est quoi ISI ?"


@requires_supabase
def test_process_pdf_insere_chunks_et_embeddings_en_base():
    async def scenario() -> None:
        document_id = None
        try:
            resultat = await process_pdf(PDF_REEL)
            document_id = resultat["document_id"]

            assert resultat["nb_chunks"] >= 1
            assert len(resultat["chunk_ids"]) == resultat["nb_chunks"]

            pool = await get_pool()
            async with pool.acquire() as conn:
                dims = await conn.fetch(
                    "SELECT vector_dims(embedding) AS dim FROM chunks WHERE document_id = $1",
                    document_id,
                )

            assert len(dims) == resultat["nb_chunks"]
            assert all(row["dim"] == DIMENSION_EMBEDDING for row in dims)
        finally:
            if document_id is not None:
                pool = await get_pool()
                async with pool.acquire() as conn:
                    await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
            await close_pool()

    asyncio.run(scenario())
