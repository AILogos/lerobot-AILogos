"""Étape 1 — Enregistrer des démonstrations de saisie d'un objet.

Vous pilotez le bras suiveur avec le bras leader ; chaque épisode montre une saisie
complète de l'objet. Le dataset est enregistré sous
``<utilisateur_hf>/ailogos_attrape_<objet>``.

Exemples (depuis la racine du dépôt) ::

    uv run python -m AILogos.enregistrer --objet balle
    uv run python -m AILogos.enregistrer --objet crayon --episodes 30
    uv run python -m AILogos.enregistrer --objet balle --reprendre      # ajoute des épisodes
    uv run python -m AILogos.enregistrer --objet balle --dry-run        # affiche la commande

Tout ce qui suit ``--`` est transmis tel quel à ``lerobot-record``.
"""

from __future__ import annotations

from .cli import ajouter_choix_objets, analyseur, charger, executer, separer_supplementaires
from .commandes import afficher_commande, commande_enregistrement, images_du_dataset, lancer

CONSEILS = """
┌─ Enregistrement : « {tache} » ─────────────────────────────────────────
│ Dataset   : {repo}
│ Épisodes  : {episodes} × {duree:g} s (remise en place : {reset:g} s)
│
│ Avant de commencer
│   • bras en position de repos, objet posé dans la zone de travail ;
│   • éclairage constant, caméras fixes (ne plus les déplacer ensuite) ;
│   • 5 à 10 saisies d'échauffement sans enregistrer.
│
│ Pendant l'enregistrement
│   • →  épisode suivant      ←  refaire l'épisode      Échap  terminer
│   • même geste d'approche et même prise à chaque épisode ;
│   • commencer avec l'objet toujours au même endroit, puis varier peu à peu.
└────────────────────────────────────────────────────────────────────────
"""


def principal(argv: list[str] | None = None) -> int:
    argv, supplementaires = separer_supplementaires(argv)
    parser = analyseur(__doc__ or "")
    ajouter_choix_objets(parser, tous=False)
    parser.add_argument("--episodes", type=int, help="nombre d'épisodes (défaut : config.yaml)")
    parser.add_argument("--reprendre", action="store_true", help="ajouter des épisodes au dataset existant")
    parser.add_argument("--sans-affichage", action="store_true", help="ne pas ouvrir la fenêtre Rerun")
    parser.add_argument("--dry-run", action="store_true", help="afficher la commande sans la lancer")
    args = parser.parse_args(argv)
    cfg = charger(args)
    objet = cfg.objet(args.objet)

    existant = images_du_dataset(cfg, objet.nom)
    if existant is not None and not args.reprendre and not args.dry_run:
        print(
            f"Le dataset {cfg.repo_dataset(objet.nom)} existe déjà ({existant} images) dans\n"
            f"  {cfg.racine_dataset(objet.nom)}\n"
            "Ajoutez --reprendre pour y ajouter des épisodes, ou supprimez ce dossier pour repartir de zéro."
        )
        return 1
    if existant is None and args.reprendre:
        print(f"Rien à reprendre : aucun dataset local pour « {objet.nom} ». Relancez sans --reprendre.")
        return 1

    commande = commande_enregistrement(
        cfg,
        objet.nom,
        episodes=args.episodes,
        reprendre=args.reprendre,
        afficher=not args.sans_affichage,
        supplementaires=supplementaires,
    )
    if args.dry_run:
        print(afficher_commande(commande))
        return 0

    enr = cfg.enregistrement
    print(
        CONSEILS.format(
            tache=objet.tache,
            repo=cfg.repo_dataset(objet.nom),
            episodes=args.episodes or enr.episodes,
            duree=enr.duree_episode_s,
            reset=enr.duree_reset_s,
        )
    )
    code = lancer(commande)
    if code == 0:
        print(
            f"\nEnregistrement terminé. Étape suivante :\n"
            f"  uv run python -m AILogos.entrainer --objet {objet.nom}"
        )
    return code


if __name__ == "__main__":
    executer(principal)
