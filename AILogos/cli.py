"""Petits utilitaires communs aux programmes en ligne de commande AILogos."""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import threading
from collections.abc import Callable

from .configuration import CHEMIN_CONFIG_DEFAUT, Configuration, ErreurConfiguration, charger_configuration


def analyseur(description: str) -> argparse.ArgumentParser:
    """Analyseur d'arguments avec l'option ``--config`` commune à tous les programmes."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        default=str(CHEMIN_CONFIG_DEFAUT),
        help="fichier de configuration (défaut : AILogos/config.yaml)",
    )
    return parser


def ajouter_choix_objets(parser: argparse.ArgumentParser, *, tous: bool = True) -> None:
    groupe = parser.add_mutually_exclusive_group(required=True)
    groupe.add_argument("--objet", help="objet concerné, tel que défini dans config.yaml (ex. balle)")
    if tous:
        groupe.add_argument("--tous", action="store_true", help="tous les objets de config.yaml")


def objets_choisis(cfg: Configuration, args: argparse.Namespace) -> list[str]:
    if getattr(args, "tous", False):
        return list(cfg.objets)
    return [cfg.objet(args.objet).nom]


def separer_supplementaires(argv: list[str] | None) -> tuple[list[str], list[str]]:
    """Sépare ``... -- --option.lerobot=valeur`` : la partie après ``--`` est transmise à LeRobot."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:
        position = argv.index("--")
        return argv[:position], argv[position + 1 :]
    return argv, []


def executer(principal: Callable[[], int | None]) -> None:
    """Lance ``principal`` en affichant les erreurs de configuration sans trace Python."""
    try:
        code = principal()
    except ErreurConfiguration as erreur:
        print(f"\n[configuration] {erreur}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("\nInterrompu.")
        code = 130
    sys.exit(code or 0)


def charger(args: argparse.Namespace) -> Configuration:
    return charger_configuration(args.config)


def oui_non(question: str, defaut: bool | None = None) -> bool:
    """Question fermée dans le terminal ; ``defaut`` est la réponse sur simple Entrée."""
    suffixe = {None: "[o/n]", True: "[O/n]", False: "[o/N]"}[defaut]
    while True:
        reponse = input(f"{question} {suffixe} ").strip().lower()
        if not reponse and defaut is not None:
            return defaut
        if reponse in ("o", "oui", "y", "yes"):
            return True
        if reponse in ("n", "non", "no"):
            return False


class EcouteEntree:
    """Déclenche ``evenement`` quand l'utilisateur appuie sur Entrée, sans bloquer le programme.

    S'utilise avec ``with`` autour d'une action longue (une saisie du robot). La surveillance
    s'arrête proprement à la sortie du bloc : la touche ne « mange » pas la question suivante.
    Disponible sur Linux et macOS dans un terminal ; ailleurs, seul Ctrl+C interrompt.
    """

    def __init__(self, evenement: threading.Event):
        self.evenement = evenement
        self._fin = threading.Event()
        self._fil: threading.Thread | None = None

    @staticmethod
    def disponible() -> bool:
        return os.name == "posix" and sys.stdin is not None and sys.stdin.isatty()

    def _surveiller(self) -> None:
        while not self._fin.is_set():
            pret, _, _ = select.select([sys.stdin], [], [], 0.1)
            if pret:
                sys.stdin.readline()
                self.evenement.set()
                return

    def __enter__(self) -> EcouteEntree:
        if self.disponible():
            self._fil = threading.Thread(target=self._surveiller, daemon=True)
            self._fil.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._fin.set()
        if self._fil is not None:
            self._fil.join(timeout=1.0)


def configurer_journalisation(verbeux: bool) -> None:
    """Par défaut seuls les avertissements de LeRobot s'affichent ; ``--verbeux`` montre tout."""
    logging.basicConfig(
        level=logging.INFO if verbeux else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
