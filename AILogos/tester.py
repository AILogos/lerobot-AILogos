"""Étape 3 — Tester la politique sur le robot et mesurer son taux de réussite.

Le programme enchaîne des essais : vous placez l'objet, le robot tente la saisie,
vous notez le résultat (réussi / raté). Chaque essai est consigné dans
``outputs/ailogos/journal.jsonl`` ; ``monitorer.py bilan`` en fait la synthèse.

Exemples (depuis la racine du dépôt) ::

    uv run python -m AILogos.tester --objet balle
    uv run python -m AILogos.tester --tous --essais 5
    uv run python -m AILogos.tester --objet balle --afficher        # caméras et actions dans Rerun
    uv run python -m AILogos.tester --objet balle --simulation      # sans matériel : la politique
                                                                    # tourne sur un robot simulé

Pendant un essai : Entrée termine l'essai (objet attrapé ou échec évident), Ctrl+C aussi.
Dans tous les cas le bras revient à sa position de départ puis est relâché.
"""

from __future__ import annotations

import contextlib
import threading

from . import journal
from .cli import (
    EcouteEntree,
    ajouter_choix_objets,
    analyseur,
    charger,
    configurer_journalisation,
    executer,
    objets_choisis,
)
from .configuration import Configuration
from .moteur import ErreurExecution, MoteurSaisie, ResultatExecution

SEUIL_BON_TAUX = 0.7


def _demander_note() -> tuple[bool | None, str]:
    """Demande le résultat de l'essai : (réussite ou ``None`` si l'essai est ignoré, commentaire)."""
    while True:
        reponse = input("  Résultat ? [o] réussi  [n] raté  [i] ignorer l'essai : ").strip().lower()
        if reponse in ("o", "oui", "y"):
            return True, ""
        if reponse in ("i", "ignorer"):
            return None, ""
        if reponse in ("n", "non"):
            commentaire = input("  Que s'est-il passé ? (facultatif, Entrée pour passer) : ").strip()
            return False, commentaire


def _afficher_sante(resultat: ResultatExecution) -> None:
    sante = resultat.sante_apres or resultat.sante_avant
    if sante and sante.get("niveau") != "ok":
        print(f"  ⚠ Moteurs ({sante['niveau']}) : " + " ; ".join(sante.get("messages", [])))


def tester_objet(
    cfg: Configuration,
    moteur: MoteurSaisie,
    objet: str,
    essais: int,
    duree_s: float,
    *,
    noter: bool = True,
) -> list[bool]:
    """Enchaîne ``essais`` tentatives de saisie de ``objet`` ; retourne les notes obtenues."""
    notes: list[bool] = []
    aide_fin = "Entrée ou Ctrl+C" if EcouteEntree.disponible() else "Ctrl+C"
    for numero in range(1, essais + 1):
        print(f"\n── {objet} · essai {numero}/{essais} ─────────────────────────────")
        if noter:
            reponse = input("  Placez l'objet, bras au repos. Entrée pour lancer, q pour arrêter : ")
            if reponse.strip().lower() in ("q", "quitter"):
                break
        print(f"  Saisie en cours ({duree_s:g} s max) — {aide_fin} pour terminer l'essai…")

        arret = threading.Event()
        with EcouteEntree(arret) if noter else contextlib.nullcontext():
            resultat = moteur.executer(objet, duree_s, arret)
        print(f"  Terminé en {resultat.duree_s:.1f} s" + (" (arrêt demandé)" if resultat.interrompu else ""))
        _afficher_sante(resultat)
        if resultat.erreur:
            print(f"  ✗ Erreur pendant l'exécution : {resultat.erreur}")

        succes, commentaire = (None, "") if resultat.erreur else (_demander_note() if noter else (None, ""))
        if succes is not None:
            notes.append(succes)
        journal.ecrire(
            cfg.journal_essais,
            "essai",
            source="test",
            objet=objet,
            succes=succes,
            commentaire=commentaire,
            duree_s=round(resultat.duree_s, 2),
            interrompu=resultat.interrompu,
            simulation=resultat.simule,
            erreur=resultat.erreur,
            politique=cfg.chemin_politique(objet),
            sante=resultat.sante_apres,
        )
    return notes


def _resume(objet: str, notes: list[bool]) -> None:
    if not notes:
        print(f"\n{objet} : aucun essai noté.")
        return
    reussites = sum(notes)
    taux = reussites / len(notes)
    print(f"\n{objet} : {reussites}/{len(notes)} saisies réussies ({taux:.0%})")
    if taux < SEUIL_BON_TAUX:
        print(
            "  Pistes : enregistrer 10 à 20 démonstrations ciblées sur les cas d'échec\n"
            f"  (uv run python -m AILogos.enregistrer --objet {objet} --reprendre), puis réentraîner ;\n"
            "  vérifier aussi l'éclairage et le cadrage des caméras."
        )


def principal(argv: list[str] | None = None) -> int:
    parser = analyseur(__doc__ or "")
    ajouter_choix_objets(parser)
    parser.add_argument("--essais", type=int, help="nombre d'essais par objet (défaut : config.yaml)")
    parser.add_argument("--duree", type=float, help="durée maximale d'un essai en secondes")
    parser.add_argument("--simulation", action="store_true", help="aucun matériel : robot simulé")
    parser.add_argument("--afficher", action="store_true", help="visualiser caméras et actions dans Rerun")
    parser.add_argument(
        "--sans-note", action="store_true", help="enchaîner les essais sans rien demander (rodage)"
    )
    parser.add_argument("--verbeux", action="store_true", help="afficher les journaux détaillés de LeRobot")
    args = parser.parse_args(argv)
    cfg = charger(args)
    configurer_journalisation(args.verbeux)

    moteur = MoteurSaisie(cfg, simulation=args.simulation, afficher=args.afficher)
    objets = objets_choisis(cfg, args)
    try:
        for objet in objets:
            moteur.verifier(objet)
    except ErreurExecution as erreur:
        print(erreur)
        return 1

    essais = args.essais or cfg.test.essais
    duree_s = args.duree or cfg.execution.duree_s
    if args.simulation:
        print("Mode simulation : aucun moteur ne bouge, les images sont du bruit.")
    try:
        for objet in objets:
            notes = tester_objet(cfg, moteur, objet, essais, duree_s, noter=not args.sans_note)
            _resume(objet, notes)
    except ErreurExecution as erreur:
        print(f"\n{erreur}")
        return 1
    finally:
        moteur.fermer()
    print(f"\nJournal : {cfg.journal_essais}\nSynthèse : uv run python -m AILogos.monitorer bilan")
    return 0


if __name__ == "__main__":
    executer(principal)
