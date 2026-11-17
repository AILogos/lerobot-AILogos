"""Étape 2 — Entraîner la politique de saisie d'un objet (ACT par défaut).

Une politique est entraînée par objet, à partir du dataset enregistré à l'étape 1.
Les checkpoints vont dans ``outputs/train/ailogos_act_<objet>/`` et la sortie de
l'entraînement est recopiée dans ``outputs/ailogos/entrainement_<objet>.log``, que
``monitorer.py entrainement`` lit pour afficher la progression.

Exemples (depuis la racine du dépôt) ::

    uv run python -m AILogos.entrainer --objet balle
    uv run python -m AILogos.entrainer --tous
    uv run python -m AILogos.entrainer --objet balle --reprendre          # après une coupure
    uv run python -m AILogos.entrainer --objet balle --steps 60000 --device cpu
    uv run python -m AILogos.entrainer --objet balle -- --job.target=a10g-small   # GPU distant (HF Jobs)

Tout ce qui suit ``--`` est transmis tel quel à ``lerobot-train``.
"""

from __future__ import annotations

from .cli import ajouter_choix_objets, analyseur, charger, executer, objets_choisis, separer_supplementaires
from .commandes import afficher_commande, commande_entrainement, images_du_dataset, lancer, nombre_de_pas
from .configuration import Configuration


def _entrainer(cfg: Configuration, objet: str, args, supplementaires: list[str]) -> int:
    sortie = cfg.dossier_entrainement(objet)
    dernier = sortie / "checkpoints" / "last"

    if args.reprendre and not dernier.exists():
        print(f"[{objet}] rien à reprendre : aucun checkpoint dans {sortie}. Relancez sans --reprendre.")
        return 1
    if not args.reprendre and sortie.exists() and not args.dry_run:
        print(
            f"[{objet}] un entraînement existe déjà dans {sortie}.\n"
            "  --reprendre pour le continuer, ou supprimez ce dossier pour repartir de zéro."
        )
        return 1
    distant = any(argument.startswith("--job.target") for argument in supplementaires)
    if images_du_dataset(cfg, objet) is None and not cfg.envoyer_sur_hub and not distant and not args.dry_run:
        print(
            f"[{objet}] dataset introuvable : {cfg.racine_dataset(objet)}\n"
            f"  Enregistrez d'abord des démonstrations : uv run python -m AILogos.enregistrer --objet {objet}"
        )
        return 1

    commande = commande_entrainement(
        cfg,
        objet,
        steps=args.steps,
        device=args.device,
        reprendre=args.reprendre,
        supplementaires=supplementaires,
    )
    if args.dry_run:
        print(afficher_commande(commande))
        return 0

    if not args.reprendre:
        pas, origine = (args.steps, "option --steps") if args.steps else nombre_de_pas(cfg, objet)
        print(f"[{objet}] entraînement {cfg.entrainement.politique} : {pas} pas ({origine})")
    print(f"[{objet}] journal : {cfg.journal_entrainement(objet)}")
    print(f"[{objet}] suivi   : uv run python -m AILogos.monitorer entrainement --objet {objet} --suivre\n")

    code = lancer(commande, journal=cfg.journal_entrainement(objet))
    if code == 0:
        print(
            f"\n[{objet}] entraînement terminé : {cfg.chemin_politique(objet)}\n"
            f"Étape suivante : uv run python -m AILogos.tester --objet {objet}"
        )
    else:
        print(f"\n[{objet}] l'entraînement s'est arrêté avec le code {code}.")
    return code


def principal(argv: list[str] | None = None) -> int:
    argv, supplementaires = separer_supplementaires(argv)
    parser = analyseur(__doc__ or "")
    ajouter_choix_objets(parser)
    parser.add_argument("--steps", type=int, help="nombre de pas (défaut : calculé depuis config.yaml)")
    parser.add_argument("--device", help="cuda, mps ou cpu (défaut : config.yaml)")
    parser.add_argument("--reprendre", action="store_true", help="reprendre depuis le dernier checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="afficher la commande sans la lancer")
    args = parser.parse_args(argv)
    cfg = charger(args)

    for objet in objets_choisis(cfg, args):
        code = _entrainer(cfg, objet, args, supplementaires)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    executer(principal)
