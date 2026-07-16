"""Harnais d'evaluation : appelle POST /chat pour chaque question de
data/eval_questions.jsonl et consigne les resultats bruts (question,
reponse, sources, temps, comportement attendu vs observe) dans
reports/eval_results.md.

Ce script ne rend aucun verdict de qualite automatique : il produit les
donnees brutes necessaires a une validation manuelle.

Prerequis : l'API doit deja tourner (`uvicorn main:app`) avec la variable
LLM_PROVIDER souhaitee pour ce run (groq ou ollama).

Usage :
    python tests/eval/run_eval.py --provider-label groq
    python tests/eval/run_eval.py --provider-label ollama
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

QUESTIONS_PATH = Path("data/eval_questions.jsonl")
RAPPORT_PATH_PAR_DEFAUT = Path("reports/eval_results.md")
BASE_URL_PAR_DEFAUT = "http://127.0.0.1:8000"
TIMEOUT_SECONDES = 200.0

ENTETE_RAPPORT = (
    "# Evaluation RAG - chatbot ISI\n\n"
    "Donnees brutes de comparaison Groq vs Ollama, generees par "
    "`tests/eval/run_eval.py`. La validation qualitative finale est manuelle.\n"
)


def charger_questions() -> list[dict]:
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        return [json.loads(ligne) for ligne in f if ligne.strip()]


def _extraire_detail_erreur(exc: httpx.HTTPStatusError) -> str:
    """Extrait le vrai corps 'detail' renvoye par l'API plutot que le message
    generique httpx ('Server error ... for url ...') qui le masque."""
    try:
        corps = exc.response.json()
        detail = corps.get("detail", corps)
    except Exception:
        detail = exc.response.text
    return f"HTTP {exc.response.status_code} : {detail}"


def executer_evaluation(base_url: str) -> list[dict]:
    questions = charger_questions()
    resultats = []
    with httpx.Client(base_url=base_url, timeout=TIMEOUT_SECONDES) as client:
        for q in questions:
            debut = time.time()
            try:
                reponse = client.post("/chat", json={"question": q["question"]})
                reponse.raise_for_status()
                corps = reponse.json()
                resultats.append(
                    {
                        **q,
                        "reponse_observee": corps["reponse"],
                        "sources": corps["sources"],
                        "temps_ms": corps["temps_ms"],
                        "erreur": None,
                    }
                )
                print(f"  [{q['id']}] ok ({corps['temps_ms']} ms)")
            except httpx.HTTPStatusError as exc:
                temps_ms = round((time.time() - debut) * 1000)
                detail = _extraire_detail_erreur(exc)
                resultats.append(
                    {
                        **q,
                        "reponse_observee": None,
                        "sources": [],
                        "temps_ms": temps_ms,
                        "erreur": detail,
                    }
                )
                print(f"  [{q['id']}] ERREUR ({temps_ms} ms): {detail}")
            except Exception as exc:
                temps_ms = round((time.time() - debut) * 1000)
                detail = f"{type(exc).__name__}: {exc}"
                resultats.append(
                    {
                        **q,
                        "reponse_observee": None,
                        "sources": [],
                        "temps_ms": temps_ms,
                        "erreur": detail,
                    }
                )
                print(f"  [{q['id']}] ERREUR ({temps_ms} ms): {detail}")
    return resultats


def formater_source(source: dict) -> str:
    return (
        f"{source['document']} (chunk {source['chunk_id']}, "
        f"page {source['numero_page']}, distance {source['distance']})"
    )


def formater_section_markdown(provider_label: str, resultats: list[dict]) -> str:
    horodatage = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lignes = [f"## Resultats - {provider_label}", "", f"_Genere le {horodatage}_", ""]

    temps_reussis = [r["temps_ms"] for r in resultats if r["erreur"] is None]
    if temps_reussis:
        moyenne = sum(temps_reussis) / len(temps_reussis)
        lignes.append(
            f"**Temps de reponse** : moyenne {moyenne:.0f} ms, "
            f"min {min(temps_reussis)} ms, max {max(temps_reussis)} ms "
            f"({len(temps_reussis)}/{len(resultats)} requetes reussies)."
        )
        lignes.append("")

    for r in resultats:
        lignes.append(f"### {r['id']}. {r['question']}")
        lignes.append(f"- **Categorie** : {r['categorie']}")
        lignes.append(f"- **Comportement attendu** : {r['comportement_attendu']}")
        if r["erreur"]:
            lignes.append(f"- **Temps ecoule avant l'erreur** : {r['temps_ms']} ms")
            lignes.append(f"- **Erreur** : {r['erreur']}")
        else:
            lignes.append(f"- **Temps de reponse** : {r['temps_ms']} ms")
            lignes.append("- **Reponse observee** :")
            lignes.append("")
            lignes.append("  > " + r["reponse_observee"].replace("\n", "\n  > "))
            lignes.append("")
            if r["sources"]:
                sources_str = "; ".join(formater_source(s) for s in r["sources"])
                lignes.append(f"- **Sources citees** : {sources_str}")
            else:
                lignes.append("- **Sources citees** : aucune (mode degrade)")
        lignes.append("")

    return "\n".join(lignes)


def ecrire_rapport(section: str, chemin_sortie: Path) -> None:
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    if chemin_sortie.exists():
        contenu_existant = chemin_sortie.read_text(encoding="utf-8")
        if not contenu_existant.startswith("# Evaluation RAG"):
            contenu_existant = ENTETE_RAPPORT + "\n" + contenu_existant
        nouveau_contenu = contenu_existant.rstrip("\n") + "\n\n" + section + "\n"
    else:
        nouveau_contenu = ENTETE_RAPPORT + "\n" + section + "\n"

    chemin_sortie.write_text(nouveau_contenu, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harnais d'evaluation RAG (POST /chat)")
    parser.add_argument(
        "--provider-label",
        required=True,
        help="Etiquette du provider actif sur l'API en cours (ex: groq, ollama)",
    )
    parser.add_argument("--base-url", default=BASE_URL_PAR_DEFAUT)
    parser.add_argument("--output", default=str(RAPPORT_PATH_PAR_DEFAUT))
    args = parser.parse_args()

    print(f"Evaluation en cours contre {args.base_url} (provider: {args.provider_label})...")
    resultats = executer_evaluation(args.base_url)

    section = formater_section_markdown(args.provider_label, resultats)
    chemin_sortie = Path(args.output)
    ecrire_rapport(section, chemin_sortie)

    print(f"Rapport ecrit : {chemin_sortie}")


if __name__ == "__main__":
    sys.exit(main())
