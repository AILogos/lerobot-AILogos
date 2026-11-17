"""Lecture et validation de ``AILogos/config.yaml``.

Ce module ne dépend que de PyYAML : il s'importe sans torch ni matériel, ce qui
permet à l'interpréteur de commandes vocales et aux tests de tourner partout.
Les objets LeRobot (``RobotConfig``…) ne sont construits qu'à la demande.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from lerobot.robots import RobotConfig
    from lerobot.teleoperators import TeleoperatorConfig

DOSSIER_AILOGOS = Path(__file__).resolve().parent
CHEMIN_CONFIG_DEFAUT = DOSSIER_AILOGOS / "config.yaml"
PREFIXE = "ailogos"
_NOM_OBJET = re.compile(r"^[a-z0-9_]+$")


class ErreurConfiguration(ValueError):
    """Le fichier de configuration est absent, incomplet ou incohérent."""


@dataclass(frozen=True)
class Objet:
    """Un objet que le robot sait attraper."""

    nom: str
    tache: str
    synonymes: tuple[str, ...]
    politique: str | None = None  # chemin local ou repo Hub imposé dans la config


@dataclass(frozen=True)
class Enregistrement:
    episodes: int = 50
    duree_episode_s: float = 20
    duree_reset_s: float = 10
    fps: int = 30


@dataclass(frozen=True)
class Entrainement:
    politique: str = "act"
    device: str = "cuda"
    batch_size: int = 8
    steps: int | None = None
    epoques: float = 10
    steps_min: int = 20_000
    sauvegarde_tous_les: int = 10_000
    journal_tous_les: int = 100
    validation: float = 0.1
    trackio: bool = False


@dataclass(frozen=True)
class Execution:
    duree_s: float = 20
    device: str | None = None
    interpolation: int = 1


@dataclass(frozen=True)
class Test:
    essais: int = 10


@dataclass(frozen=True)
class Sante:
    temperature_alerte_c: float = 55
    temperature_critique_c: float = 65
    tension_min_v: float = 4.5
    tension_max_v: float = 13.5
    charge_alerte_pct: float = 85


@dataclass(frozen=True)
class Voix:
    langue: str = "fr"
    modele: str = "small"
    device: str = "cpu"
    precision: str = "int8"
    micro: int | str | None = None
    mot_reveil: str = ""
    seuil_parole: float = 2.5
    silence_fin_s: float = 0.8
    duree_max_s: float = 6
    retour_vocal: bool = True


@dataclass(frozen=True)
class Configuration:
    """Vue typée de ``config.yaml``."""

    utilisateur_hf: str
    envoyer_sur_hub: bool
    dossier_sorties: Path
    robot: dict[str, Any]
    teleop: dict[str, Any]
    objets: dict[str, Objet]
    enregistrement: Enregistrement = field(default_factory=Enregistrement)
    entrainement: Entrainement = field(default_factory=Entrainement)
    execution: Execution = field(default_factory=Execution)
    test: Test = field(default_factory=Test)
    sante: Sante = field(default_factory=Sante)
    voix: Voix = field(default_factory=Voix)
    chemin: Path | None = None

    # ── Objets ───────────────────────────────────────────────────────────────

    def objet(self, nom: str) -> Objet:
        try:
            return self.objets[nom]
        except KeyError:
            connus = ", ".join(self.objets)
            raise ErreurConfiguration(f"Objet inconnu : « {nom} ». Objets connus : {connus}.") from None

    # ── Noms dérivés (identiques dans tous les programmes) ───────────────────

    def nom_tache(self, objet: str) -> str:
        return f"{PREFIXE}_attrape_{self.objet(objet).nom}"

    def repo_dataset(self, objet: str) -> str:
        return f"{self.utilisateur_hf}/{self.nom_tache(objet)}"

    def racine_dataset(self, objet: str) -> Path:
        """Dossier local du dataset (là où ``lerobot-record`` l'écrit par défaut)."""
        return _dossier_lerobot() / self.repo_dataset(objet)

    def nom_entrainement(self, objet: str) -> str:
        return f"{PREFIXE}_{self.entrainement.politique}_{self.objet(objet).nom}"

    def dossier_entrainement(self, objet: str) -> Path:
        return self.dossier_sorties / "train" / self.nom_entrainement(objet)

    def repo_politique(self, objet: str) -> str:
        return f"{self.utilisateur_hf}/{self.nom_entrainement(objet)}"

    def chemin_politique(self, objet: str) -> str:
        """Politique à charger pour cet objet : celle de la config, sinon le dernier checkpoint local."""
        impose = self.objet(objet).politique
        if impose:
            return impose
        return str(self.dossier_entrainement(objet) / "checkpoints" / "last" / "pretrained_model")

    def politique_disponible(self, objet: str) -> bool:
        """Vrai si la politique est sur le disque (un identifiant Hub est supposé disponible)."""
        chemin = Path(self.chemin_politique(objet))
        if chemin.exists():
            return (chemin / "config.json").exists()
        # Un identifiant « utilisateur/depot » imposé dans la config sera téléchargé au chargement.
        return self.objet(objet).politique is not None and not chemin.is_absolute()

    @property
    def dossier_journaux(self) -> Path:
        return self.dossier_sorties / PREFIXE

    def journal_entrainement(self, objet: str) -> Path:
        return self.dossier_journaux / f"entrainement_{self.objet(objet).nom}.log"

    @property
    def journal_essais(self) -> Path:
        return self.dossier_journaux / "journal.jsonl"

    # ── Matériel ─────────────────────────────────────────────────────────────

    def arguments_robot(self) -> list[str]:
        """``--robot.*`` pour les commandes ``lerobot-*``."""
        return _arguments_cli("robot", self.robot)

    def arguments_teleop(self) -> list[str]:
        """``--teleop.*`` pour les commandes ``lerobot-*``."""
        return _arguments_cli("teleop", self.teleop)

    def config_robot(self) -> RobotConfig:
        """Construit la ``RobotConfig`` LeRobot (une instance neuve à chaque appel)."""
        import draccus

        from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  (enregistre le type « opencv »)
        from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
        from lerobot.robots import RobotConfig, koch_follower, so_follower  # noqa: F401
        from lerobot.utils import register_third_party_plugins

        from . import robot_simule  # noqa: F401  (enregistre le type « ailogos_simule »)

        register_third_party_plugins()
        return draccus.decode(RobotConfig, copy.deepcopy(self.robot))

    def config_teleop(self) -> TeleoperatorConfig:
        import draccus

        from lerobot.teleoperators import TeleoperatorConfig, koch_leader, so_leader  # noqa: F401
        from lerobot.utils import register_third_party_plugins

        register_third_party_plugins()
        return draccus.decode(TeleoperatorConfig, copy.deepcopy(self.teleop))


def _dossier_lerobot() -> Path:
    """Équivalent de ``lerobot.utils.constants.HF_LEROBOT_HOME`` sans importer lerobot."""
    if os.environ.get("HF_LEROBOT_HOME"):
        return Path(os.environ["HF_LEROBOT_HOME"]).expanduser()
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "lerobot"


def _arguments_cli(prefixe: str, section: dict[str, Any]) -> list[str]:
    """Transforme une section YAML en arguments draccus (``--robot.port=...``).

    Les valeurs composées (les caméras) sont passées en JSON, que draccus lit comme du YAML.
    Les valeurs ``null`` sont omises pour laisser les défauts de LeRobot s'appliquer.
    """
    arguments = []
    for cle, valeur in section.items():
        if valeur is None:
            continue
        if isinstance(valeur, bool):
            texte = "true" if valeur else "false"
        elif isinstance(valeur, (dict, list)):
            texte = json.dumps(valeur, ensure_ascii=False)
        else:
            texte = str(valeur)
        arguments.append(f"--{prefixe}.{cle}={texte}")
    return arguments


def _section(brut: dict[str, Any], nom: str, classe: type) -> Any:
    """Construit une dataclass depuis une section YAML, en refusant les clés inconnues."""
    valeurs = brut.get(nom) or {}
    if not isinstance(valeurs, dict):
        raise ErreurConfiguration(f"La section « {nom} » doit être un dictionnaire.")
    champs = classe.__dataclass_fields__
    inconnues = sorted(set(valeurs) - set(champs))
    if inconnues:
        raise ErreurConfiguration(
            f"Clé(s) inconnue(s) dans « {nom} » : {', '.join(inconnues)}. Clés possibles : {', '.join(champs)}."
        )
    return classe(**valeurs)


def _materiel(brut: dict[str, Any], nom: str) -> dict[str, Any]:
    section = brut.get(nom)
    if not isinstance(section, dict) or not section.get("type"):
        raise ErreurConfiguration(f"La section « {nom} » doit contenir au moins « type ».")
    return section


def _objets(brut: dict[str, Any]) -> dict[str, Objet]:
    section = brut.get("objets")
    if not isinstance(section, dict) or not section:
        raise ErreurConfiguration("La section « objets » doit définir au moins un objet.")
    objets = {}
    for nom, valeurs in section.items():
        nom = str(nom)
        if not _NOM_OBJET.match(nom):
            raise ErreurConfiguration(
                f"Nom d'objet invalide : « {nom} ». Utiliser des minuscules, chiffres et « _ » "
                "(le nom sert dans les noms de fichiers) ; les accents vont dans « synonymes »."
            )
        valeurs = valeurs or {}
        inconnues = sorted(set(valeurs) - {"tache", "synonymes", "politique"})
        if inconnues:
            raise ErreurConfiguration(f"Clé(s) inconnue(s) pour l'objet « {nom} » : {', '.join(inconnues)}.")
        synonymes = [str(s) for s in (valeurs.get("synonymes") or [])]
        if nom not in synonymes:
            synonymes.insert(0, nom)
        objets[nom] = Objet(
            nom=nom,
            tache=str(valeurs.get("tache") or f"Attrape l'objet {nom}"),
            synonymes=tuple(synonymes),
            politique=valeurs.get("politique"),
        )
    return objets


def charger_configuration(chemin: str | Path | None = None) -> Configuration:
    """Charge ``config.yaml`` (par défaut celui du dossier AILogos)."""
    chemin = Path(chemin) if chemin else CHEMIN_CONFIG_DEFAUT
    if not chemin.exists():
        raise ErreurConfiguration(f"Fichier de configuration introuvable : {chemin}")
    with open(chemin, encoding="utf-8") as fichier:
        brut = yaml.safe_load(fichier) or {}
    if not isinstance(brut, dict):
        raise ErreurConfiguration(f"{chemin} : le fichier doit contenir un dictionnaire YAML.")

    sections = {
        "utilisateur_hf",
        "envoyer_sur_hub",
        "dossier_sorties",
        "robot",
        "teleop",
        "objets",
        "enregistrement",
        "entrainement",
        "execution",
        "test",
        "sante",
        "voix",
    }
    inconnues = sorted(set(brut) - sections)
    if inconnues:
        raise ErreurConfiguration(f"Section(s) inconnue(s) dans {chemin.name} : {', '.join(inconnues)}.")

    try:
        configuration = Configuration(
            utilisateur_hf=str(brut.get("utilisateur_hf") or "local"),
            envoyer_sur_hub=bool(brut.get("envoyer_sur_hub", False)),
            dossier_sorties=Path(brut.get("dossier_sorties") or "outputs"),
            robot=_materiel(brut, "robot"),
            teleop=_materiel(brut, "teleop"),
            objets=_objets(brut),
            enregistrement=_section(brut, "enregistrement", Enregistrement),
            entrainement=_section(brut, "entrainement", Entrainement),
            execution=_section(brut, "execution", Execution),
            test=_section(brut, "test", Test),
            sante=_section(brut, "sante", Sante),
            voix=_section(brut, "voix", Voix),
            chemin=chemin,
        )
    except TypeError as erreur:
        raise ErreurConfiguration(f"{chemin} : {erreur}") from erreur

    if configuration.envoyer_sur_hub and configuration.utilisateur_hf == "local":
        raise ErreurConfiguration(
            "« envoyer_sur_hub: true » demande un vrai nom d'utilisateur Hugging Face dans « utilisateur_hf »."
        )
    if not 0 <= configuration.entrainement.validation < 1:
        raise ErreurConfiguration("« entrainement.validation » doit être compris entre 0 et 1 (exclu).")
    if configuration.execution.interpolation < 1:
        raise ErreurConfiguration("« execution.interpolation » doit valoir au moins 1.")
    return configuration
