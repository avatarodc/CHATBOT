-- Schema PgVector pour chatbot-isi-m1gl (Supabase)
-- Extension pgvector deja activee manuellement via le SQL editor Supabase :
--   CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    nom_fichier TEXT NOT NULL,
    date_upload TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Dimension 384 = sortie du modele d'embeddings paraphrase-multilingual-MiniLM-L12-v2
CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    contenu TEXT NOT NULL,
    embedding vector(384) NOT NULL
);

-- HNSW plutot qu'IVFFLAT : index utilisable des la premiere insertion (IVFFLAT
-- doit etre construit apres avoir charge des donnees representatives via son
-- parametre "lists", ce qui ne convient pas a un corpus qui grossit chunk par chunk).
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops);
