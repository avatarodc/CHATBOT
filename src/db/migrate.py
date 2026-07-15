"""Application du schema SQL (idempotent, via CREATE TABLE/INDEX IF NOT EXISTS)."""

from pathlib import Path

from src.db.connection import get_pool

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


async def apply_schema() -> None:
    """Cree les tables et l'index s'ils n'existent pas deja."""
    pool = await get_pool()
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    async with pool.acquire() as conn:
        await conn.execute(sql)
