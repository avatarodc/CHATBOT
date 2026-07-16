"""Point d'entree FastAPI : monte les routes API et gere le cycle de vie."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.db.connection import close_pool, get_pool
from src.embeddings.model import get_modele


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_modele()  # charge le modele d'embeddings une seule fois, au demarrage
    await get_pool()  # etablit le pool DB au demarrage (echoue vite si mal configure)
    yield
    await close_pool()


app = FastAPI(title="Chatbot ISI - RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Monte apres les routes API : les chemins specifiques (/chat, /health, ...)
# sont apparies en priorite, ce mount ne sert que ce qu'ils ne couvrent pas.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
