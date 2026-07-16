"""Routes FastAPI : upload de PDF, question RAG, health check."""

import os
import uuid
from pathlib import Path

import groq
import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.api.rag import answer_question
from src.db.connection import DatabaseConfigError, DatabaseConnectionError, get_pool
from src.embeddings.pipeline import process_pdf
from src.ingestion.extraction import PDFCorrompuError, PDFSansTexteExtractibleError
from src.llm.base import LLMProviderError
from src.llm.ollama_provider import OLLAMA_URL_PAR_DEFAUT

router = APIRouter()

DOSSIER_UPLOADS = Path("data/uploads")
TAILLE_MAX_OCTETS = 20 * 1024 * 1024  # 20 Mo

MESSAGE_ERREUR_INTERNE = "Erreur interne. Reessayez plus tard ou contactez l'administrateur."


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceItem(BaseModel):
    document: str
    chunk_id: int
    numero_page: int | None
    distance: float


class ChatResponse(BaseModel):
    reponse: str
    sources: list[SourceItem]
    temps_ms: int


class UploadResponse(BaseModel):
    document_id: int
    nom_fichier: str
    nb_chunks: int


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    nom_fichier = (file.filename or "").lower()
    if file.content_type != "application/pdf" or not nom_fichier.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Seuls les fichiers PDF sont acceptes.")

    contenu = await file.read()
    if len(contenu) > TAILLE_MAX_OCTETS:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (max {TAILLE_MAX_OCTETS // (1024 * 1024)} Mo).",
        )

    DOSSIER_UPLOADS.mkdir(parents=True, exist_ok=True)
    nom_sur_disque = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    chemin = DOSSIER_UPLOADS / nom_sur_disque
    chemin.write_bytes(contenu)

    try:
        resultat = await process_pdf(str(chemin))
    except (PDFCorrompuError, PDFSansTexteExtractibleError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except (DatabaseConfigError, DatabaseConnectionError, LLMProviderError):
        raise HTTPException(status_code=500, detail=MESSAGE_ERREUR_INTERNE) from None
    except Exception:
        raise HTTPException(status_code=500, detail=MESSAGE_ERREUR_INTERNE) from None

    return UploadResponse(
        document_id=resultat["document_id"],
        nom_fichier=resultat["nom_fichier"],
        nb_chunks=resultat["nb_chunks"],
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(requete: QuestionRequest) -> ChatResponse:
    try:
        resultat = await answer_question(requete.question)
    except (DatabaseConfigError, DatabaseConnectionError):
        raise HTTPException(status_code=503, detail="Service de recherche indisponible.") from None
    except LLMProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except Exception:
        raise HTTPException(status_code=500, detail=MESSAGE_ERREUR_INTERNE) from None

    return ChatResponse(
        reponse=resultat["reponse"],
        sources=[SourceItem(**s) for s in resultat["sources"]],
        temps_ms=round(resultat["temps_traitement"] * 1000),
    )


@router.get("/health")
async def health() -> dict:
    db_ok = await _verifier_db()
    llm_info = await _verifier_llm()
    statut_global = "ok" if db_ok and llm_info["status"] == "up" else "degraded"
    return {"status": statut_global, "db": "up" if db_ok else "down", "llm": llm_info}


async def _verifier_db() -> bool:
    # Frontiere du systeme : le health check ne doit jamais lever d'exception,
    # seulement rapporter un statut, d'ou l'except Exception volontairement large.
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True
    except Exception:
        return False


async def _verifier_llm() -> dict:
    nom_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    if nom_provider == "groq":
        try:
            client = groq.AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", ""), timeout=5.0)
            await client.models.list()
            return {"provider": "groq", "status": "up"}
        except Exception:
            return {"provider": "groq", "status": "down"}

    if nom_provider == "ollama":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                reponse = await client.get(f"{OLLAMA_URL_PAR_DEFAUT}/api/tags")
                reponse.raise_for_status()
            return {"provider": "ollama", "status": "up"}
        except Exception:
            return {"provider": "ollama", "status": "down"}

    return {"provider": nom_provider or None, "status": "inconnu"}
