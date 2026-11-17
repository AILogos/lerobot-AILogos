"""Robot simulé : mêmes moteurs et mêmes caméras qu'un SO-101, sans aucun matériel.

Il sert à trois choses :

* ``--simulation`` dans ``tester.py`` et ``v2_commande_vocale.py`` : vérifier qu'une
  politique entraînée se charge et tourne à la bonne cadence avant d'alimenter le bras ;
* les tests automatiques du dossier ``AILogos/tests`` ;
* essayer la chaîne vocale complète sur un PC sans robot.

Les images sont du bruit : la politique « bouge » le bras virtuel sans rien attraper.
LeRobot instancie ``RobotSimule`` à partir de ``RobotSimuleConfig`` par convention de nom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots import Robot, RobotConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

MOTEURS_SO101 = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
TYPE_ROBOT_SIMULE = "ailogos_simule"


@RobotConfig.register_subclass(TYPE_ROBOT_SIMULE)
@dataclass
class RobotSimuleConfig(RobotConfig):
    # Nom de caméra → [hauteur, largeur] des images simulées.
    cameras_simulees: dict[str, list[int]] = field(
        default_factory=lambda: {"front": [480, 640], "wrist": [480, 640]}
    )
    moteurs: list[str] = field(default_factory=lambda: list(MOTEURS_SO101))
    # Part du chemin vers la consigne parcourue à chaque pas (1 = suivi instantané).
    suivi: float = 0.5


class RobotSimule(Robot):
    """Bras virtuel dont les articulations rejoignent progressivement la consigne reçue."""

    config_class = RobotSimuleConfig
    name = TYPE_ROBOT_SIMULE
    supports_position_hold = True

    def __init__(self, config: RobotSimuleConfig):
        super().__init__(config)
        self.config = config
        self._connecte = False
        self._positions = dict.fromkeys(config.moteurs, 0.0)
        self._generateur = np.random.default_rng(0)
        self.actions_recues = 0

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{moteur}.pos": float for moteur in self.config.moteurs}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        cameras = {nom: (h, w, 3) for nom, (h, w) in self.config.cameras_simulees.items()}
        return {**self.action_features, **cameras}

    @property
    def is_connected(self) -> bool:
        return self._connecte

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._connecte = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        observation: RobotObservation = {f"{moteur}.pos": pos for moteur, pos in self._positions.items()}
        for nom, (hauteur, largeur) in self.config.cameras_simulees.items():
            observation[nom] = self._generateur.integers(0, 256, (hauteur, largeur, 3), dtype=np.uint8)
        return observation

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        for cle, consigne in action.items():
            moteur = cle.removesuffix(".pos")
            if moteur in self._positions:
                actuelle = self._positions[moteur]
                self._positions[moteur] = actuelle + self.config.suivi * (float(consigne) - actuelle)
        self.actions_recues += 1
        return action

    @check_if_not_connected
    def disconnect(self) -> None:
        self._connecte = False


def section_robot_simule(section_robot: dict) -> dict:
    """Section ``robot`` simulée qui reprend les noms et tailles des caméras du vrai robot."""
    cameras = section_robot.get("cameras") or {}
    simulees = {
        nom: [int(camera.get("height", 480)), int(camera.get("width", 640))]
        for nom, camera in cameras.items()
    }
    section = {"type": TYPE_ROBOT_SIMULE, "id": "ailogos_simule"}
    if simulees:
        section["cameras_simulees"] = simulees
    return section
