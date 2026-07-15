"""Interface commune (Strategy) pour les fournisseurs LLM interchangeables."""

from abc import ABC, abstractmethod

SYSTEME_PAR_DEFAUT = (
    "Tu es un assistant qui repond aux questions en te basant uniquement sur "
    "le contexte fourni. Si le contexte ne contient pas la reponse, dis-le clairement."
)


class LLMProviderError(Exception):
    """Erreur lors de l'appel au fournisseur LLM (quota, connexion, timeout, cle invalide)."""


class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, context: list[str]) -> str:
        raise NotImplementedError

    @staticmethod
    def _construire_messages(prompt: str, context: list[str]) -> list[dict]:
        """Construit la liste de messages chat (systeme + contexte + question)."""
        if context:
            bloc_contexte = "\n\n".join(f"- {extrait}" for extrait in context)
            message_utilisateur = f"Contexte :\n{bloc_contexte}\n\nQuestion : {prompt}"
        else:
            message_utilisateur = prompt

        return [
            {"role": "system", "content": SYSTEME_PAR_DEFAUT},
            {"role": "user", "content": message_utilisateur},
        ]
