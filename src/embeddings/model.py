"""Chargement singleton du modele d'embeddings (sentence-transformers)."""

from sentence_transformers import SentenceTransformer

NOM_MODELE = "paraphrase-multilingual-MiniLM-L12-v2"
DIMENSION_EMBEDDING = 384

_modele: SentenceTransformer | None = None


def get_modele() -> SentenceTransformer:
    """Retourne l'instance unique du modele, en la chargeant si necessaire."""
    global _modele
    if _modele is None:
        _modele = SentenceTransformer(NOM_MODELE)
    return _modele


def normaliser_pour_embedding(texte: str) -> str:
    """Normalise un texte avant calcul d'embedding (minuscules).

    Un meme contenu ne doit pas produire un vecteur different selon la casse
    saisie par l'utilisateur ou presente dans le document source - sinon la
    distance de similarite peut varier au-dessus ou en-dessous du seuil pour
    un cas quasi identique. A appeler avant chaque appel a encoder(), aux
    deux endroits ou l'application calcule un embedding (ingestion des
    chunks, embedding de la question utilisateur).
    """
    return texte.lower()


def encoder(textes: list[str]) -> list[list[float]]:
    """Encode une liste de textes en embeddings de dimension DIMENSION_EMBEDDING."""
    vecteurs = get_modele().encode(textes, convert_to_numpy=True)
    return vecteurs.tolist()
