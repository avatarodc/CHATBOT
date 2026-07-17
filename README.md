# chatbot-isi-m1gl

Chatbot RAG (Retrieval-Augmented Generation) pour le Groupe ISI. Le système ingère des documents PDF (plaquettes de formations, catalogues), les indexe par recherche sémantique, et répond aux questions des utilisateurs en se basant uniquement sur le contenu de ces documents. La génération de réponse passe par un LLM configurable (Groq ou Ollama).

Pipeline : upload de PDF → extraction → chunking → embeddings → stockage PostgreSQL/pgvector (Supabase) → recherche sémantique → réponse via LLM.

## Prérequis

- Python 3.11+
- Un projet [Supabase](https://supabase.com) (PostgreSQL managé, avec l'extension pgvector activée)
- Une clé API [Groq](https://console.groq.com) et/ou [Ollama](https://ollama.com) installé localement

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
3. Recuperer l'URL de connexion PostgreSQL (Project Settings → Database → Connection string — utiliser de préférence le **connection pooler**, pas la connexion directe) et la renseigner dans `.env` sous `SUPABASE_DB_URL`.

## Configuration `.env`

Aucune valeur ci-dessous n'est une vraie clé ou un vrai mot de passe — ce sont des noms de variables à renseigner vous-même dans votre propre `.env` (jamais commité).

| Variable | Description |
|---|---|
| `SUPABASE_DB_URL` | `<connection string PostgreSQL fournie par Supabase, pgvector activé>` |
| `GROQ_API_KEY` | `<clé API Groq>`, requise si `LLM_PROVIDER=groq` |
| `LLM_PROVIDER` | Fournisseur LLM actif : `groq` ou `ollama` |
| `OLLAMA_MODEL` | Modèle Ollama à utiliser si `LLM_PROVIDER=ollama` (ex. `qwen2.5:7b-instruct-q4_K_M`) |

Ne jamais committer `.env` (déjà exclu via `.gitignore`) ni de clé réelle dans le dépôt.

## Structure du projet

```
src/
  api/         routes FastAPI (endpoints d'upload et de question) + orchestration RAG
  ingestion/   extraction PDF et chunking
  embeddings/  generation des embeddings (sentence-transformers)
  llm/         interface commune Groq / Ollama (Strategy)
  db/          acces PostgreSQL / pgvector
data/uploads/  fichiers PDF uploades
tests/         tests automatises
frontend/      page de démo statique (servie par l'API)
```

## Lancer le serveur

```powershell
uvicorn main:app --reload
```

Par défaut, le serveur écoute sur `http://127.0.0.1:8000`.

## Documentation interactive

Une fois le serveur lancé, la documentation Swagger interactive est disponible sur **`http://127.0.0.1:8000/docs`** (et la spécification OpenAPI brute sur `/openapi.json`). C'est le moyen le plus rapide d'explorer et de tester les endpoints sans lire le code source.

## CORS

Le CORS est déjà activé côté serveur (toutes origines, toutes méthodes, tous en-têtes autorisés). Un frontend en développement local (React, Vue, page statique, etc.), servi depuis n'importe quel port, peut appeler l'API directement sans configuration supplémentaire.

## Documentation des endpoints

### `GET /health`

Vérifie que la base de données et le fournisseur LLM configuré répondent. Ne prend aucun paramètre. **Retourne toujours un statut HTTP 200** — un problème est signalé dans le corps de la réponse (`status: "degraded"`), pas par un code d'erreur HTTP.

**Réponse (200)**

```json
{
  "status": "ok",
  "db": "up",
  "llm": {
    "provider": "groq",
    "status": "up"
  }
}
```

- `status` : `"ok"` si tout fonctionne, `"degraded"` sinon.
- `db` : `"up"` ou `"down"`.
- `llm.provider` : `"groq"`, `"ollama"`, ou `null` si `LLM_PROVIDER` n'est pas configuré.
- `llm.status` : `"up"`, `"down"`, ou `"inconnu"` si `LLM_PROVIDER` a une valeur inattendue.

### `POST /documents/upload`

Upload et indexation d'un PDF (extraction → chunking → embeddings → stockage). Requête `multipart/form-data` avec un champ `file`.

**Requête (exemple curl)**

```bash
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@plaquette.pdf;type=application/pdf"
```

**Réponse (200)**

```json
{
  "document_id": 12,
  "nom_fichier": "a1b2c3d4_plaquette.pdf",
  "nb_chunks": 8
}
```

Le `nom_fichier` retourné n'est pas exactement celui envoyé : un préfixe aléatoire est ajouté côté serveur pour éviter les collisions entre uploads (`a1b2c3d4_` dans l'exemple).

**Erreurs possibles**

| Code | Cas | Corps de la réponse |
|---|---|---|
| `422` | Fichier envoyé n'est pas un PDF (type ou extension) | `{"detail": "Seuls les fichiers PDF sont acceptes."}` |
| `422` | PDF corrompu / illisible | `{"detail": "Le fichier PDF '...' est illisible ou corrompu."}` |
| `422` | PDF scanné (image) sans texte extractible — l'OCR n'est pas pris en charge | `{"detail": "Aucun texte extractible dans '...'. ..."}` |
| `413` | Fichier de plus de 20 Mo | `{"detail": "Fichier trop volumineux (max 20 Mo)."}` |
| `500` | Erreur interne (base de données, configuration, ou imprévue) | `{"detail": "Erreur interne. Reessayez plus tard ou contactez l'administrateur."}` |

### `POST /chat`

Pose une question ; le serveur recherche les passages pertinents dans les documents indexés et génère une réponse via le LLM configuré.

**Requête**

```json
{
  "question": "Quels sont les campus du Groupe ISI ?"
}
```

`question` : chaîne non vide, 2000 caractères maximum.

**Réponse (200)**

```json
{
  "reponse": "Le Groupe ISI est présent sur plusieurs campus : ...",
  "sources": [
    {
      "document": "a1b2c3d4_plaquette.pdf",
      "chunk_id": 42,
      "numero_page": 1,
      "distance": 0.23
    }
  ],
  "temps_ms": 1523
}
```

- `sources` : liste des passages utilisés pour construire la réponse. Peut être **vide** (`[]`) si aucun document indexé n'est pertinent pour la question, ou pour une simple salutation — dans ce cas `reponse` contient un message d'accueil ou d'orientation plutôt qu'une réponse basée sur un document.
- `sources[].numero_page` peut être `null` si l'information n'est pas disponible.
- `distance` : distance cosinus entre la question et le passage (plus c'est bas, plus c'est pertinent).
- `temps_ms` : temps de traitement côté serveur, en millisecondes.

**Erreurs possibles**

| Code | Cas | Corps de la réponse |
|---|---|---|
| `422` | `question` vide, absente, ou trop longue (validation de schéma) | `{"detail": [{"type": "string_too_short", "loc": ["body", "question"], "msg": "String should have at least 1 character", ...}]}` |
| `503` | Base de données ou fournisseur LLM indisponible (quota dépassé, clé invalide, timeout, service non démarré) | `{"detail": "..."}` (message explicite selon la cause) |
| `500` | Erreur interne imprévue | `{"detail": "Erreur interne. Reessayez plus tard ou contactez l'administrateur."}` |

Notez que pour l'erreur `422` ci-dessus, `detail` est une **liste d'objets** (format de validation Pydantic/FastAPI), alors que pour toutes les autres erreurs de cette API, `detail` est une **chaîne de caractères**.

## Tests

```powershell
pytest
```
