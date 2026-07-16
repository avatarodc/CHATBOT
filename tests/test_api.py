"""Tests des routes FastAPI : upload PDF, chat, health."""

import asyncio
import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from src.db.connection import close_pool, get_pool

requires_supabase = pytest.mark.skipif(
    not os.environ.get("SUPABASE_DB_URL"),
    reason="SUPABASE_DB_URL non configuree dans l'environnement",
)


def test_upload_refuse_un_fichier_non_pdf():
    with TestClient(app) as client:
        reponse = client.post(
            "/documents/upload",
            files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")},
        )

    assert reponse.status_code == 422
    assert "PDF" in reponse.json()["detail"]


def test_upload_refuse_un_fichier_trop_volumineux():
    gros_contenu = b"0" * (21 * 1024 * 1024)
    with TestClient(app) as client:
        reponse = client.post(
            "/documents/upload",
            files={"file": ("gros.pdf", io.BytesIO(gros_contenu), "application/pdf")},
        )

    assert reponse.status_code == 413


def test_chat_refuse_une_question_vide():
    with TestClient(app) as client:
        reponse = client.post("/chat", json={"question": ""})

    assert reponse.status_code == 422


@requires_supabase
def test_health_repond():
    with TestClient(app) as client:
        reponse = client.get("/health")

    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["status"] in ("ok", "degraded")
    assert "db" in corps and "llm" in corps


@requires_supabase
def test_upload_puis_chat_bout_en_bout():
    document_id = None
    nom_sur_disque = None
    try:
        with TestClient(app) as client:
            with open("data/uploads/ISI_Formations_Test.pdf", "rb") as f:
                reponse_upload = client.post(
                    "/documents/upload",
                    files={"file": ("ISI_Formations_Test.pdf", f, "application/pdf")},
                )
            assert reponse_upload.status_code == 200
            corps_upload = reponse_upload.json()
            assert corps_upload["nb_chunks"] >= 1
            document_id = corps_upload["document_id"]
            nom_sur_disque = corps_upload["nom_fichier"]

            reponse_chat = client.post(
                "/chat", json={"question": "Quels sont les departements du Groupe ISI ?"}
            )
            assert reponse_chat.status_code == 200
            corps_chat = reponse_chat.json()
            assert corps_chat["reponse"].strip()
            assert corps_chat["sources"]
            assert corps_chat["temps_ms"] > 0
    finally:
        if document_id is not None:
            async def _nettoyer():
                pool = await get_pool()
                async with pool.acquire() as conn:
                    await conn.execute("DELETE FROM documents WHERE id = $1", document_id)
                await close_pool()

            asyncio.run(_nettoyer())
        if nom_sur_disque:
            (Path("data/uploads") / nom_sur_disque).unlink(missing_ok=True)
