"""Connexion asynchrone a la base Supabase (PostgreSQL + pgvector)."""

import os

import asyncpg
from dotenv import load_dotenv
from pgvector.asyncpg import register_vector

load_dotenv()


class DatabaseConfigError(Exception):
    """SUPABASE_DB_URL est absente ou invalide."""


class DatabaseConnectionError(Exception):
    """La connexion a la base de donnees a echoue."""


_pool: asyncpg.Pool | None = None


async def _register_vector_codec(conn: asyncpg.Connection) -> None:
    await register_vector(conn)


async def get_pool() -> asyncpg.Pool:
    """Retourne le pool de connexions, en le creant si necessaire."""
    global _pool
    if _pool is not None:
        return _pool

    dsn = os.environ.get("SUPABASE_DB_URL")
    if not dsn:
        raise DatabaseConfigError(
            "SUPABASE_DB_URL n'est pas definie. Verifiez le fichier .env."
        )

    try:
        # statement_cache_size=0 : le pooler Supabase (Supavisor, mode transaction)
        # ne supporte pas le cache de prepared statements d'asyncpg.
        _pool = await asyncpg.create_pool(
            dsn=dsn, init=_register_vector_codec, statement_cache_size=0
        )
    except (OSError, asyncpg.PostgresError) as exc:
        raise DatabaseConnectionError(
            "Impossible de se connecter a la base de donnees Supabase. "
            "Verifiez que le service est accessible et que les identifiants sont corrects."
        ) from exc

    return _pool


async def close_pool() -> None:
    """Ferme le pool de connexions s'il est ouvert."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
