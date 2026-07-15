# chatbot-isi-m1gl

MVP de chatbot RAG (Retrieval-Augmented Generation) — cours IA Générative, M1 GL, ISI (M. Assane BA).

Upload de PDF → extraction → chunking → embeddings → stockage PostgreSQL/pgvector (Supabase) → recherche semantique → reponse via LLM (Groq ou Ollama, configurable).

## Prerequis

- Python 3.11+
- Un projet [Supabase](https://supabase.com) (PostgreSQL manage, pas de Postgres local)
- Une cle API [Groq](https://console.groq.com) et/ou [Ollama](https://ollama.com) installe localement

## Installation (Windows / PowerShell)

```powershell
# 1. Creer et activer l'environnement virtuel
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Installer les dependances
pip install -r requirements.txt

# 3. Copier le fichier d'environnement
Copy-Item .env.example .env
```

> Si l'activation echoue avec une erreur de policy d'execution, lancer une fois :
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

## Configuration Supabase (pgvector)

1. Creer un projet sur [supabase.com](https://supabase.com).
2. Dans l'editeur SQL du projet, activer l'extension pgvector :
   ```sql
   create extension if not exists vector;
   ```
3. Recuperer l'URL de connexion PostgreSQL (Project Settings → Database → Connection string) et la renseigner dans `.env` sous `SUPABASE_DB_URL`.

## Configuration `.env`

| Variable | Description |
|---|---|
| `SUPABASE_DB_URL` | URL de connexion PostgreSQL Supabase (pgvector active) |
| `GROQ_API_KEY` | Cle API Groq, requise si `LLM_PROVIDER=groq` |
| `LLM_PROVIDER` | Fournisseur LLM actif : `groq` ou `ollama` |
| `OLLAMA_MODEL` | Modele Ollama a utiliser si `LLM_PROVIDER=ollama` (ex. `qwen2.5:7b-instruct-q4_K_M`) |

Ne jamais committer `.env` (deja exclu via `.gitignore`) ni de cle reelle dans le depot.

## Structure du projet

```
src/
  api/         routes FastAPI (endpoints d'upload et de question)
  ingestion/   extraction PDF et chunking
  embeddings/  generation des embeddings (sentence-transformers)
  llm/         interface commune Groq / Ollama (Strategy)
  db/          acces PostgreSQL / pgvector
data/uploads/  fichiers PDF uploades
tests/         tests automatises
reports/       rapports / sorties d'analyse
```

## Tests

```powershell
pytest
```

## Statut

Structure initiale du projet — aucune logique metier implementee a ce stade.
