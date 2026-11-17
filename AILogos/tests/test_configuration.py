"""Chargement de config.yaml et noms dérivés."""

from __future__ import annotations

from pathlib import Path

import pytest

from AILogos.configuration import ErreurConfiguration, charger_configuration

from .conftest import ecrire_config


def test_config_livree_est_valide():
    """Le config.yaml du dossier AILogos doit toujours se charger tel quel."""
    cfg = charger_configuration()
    assert set(cfg.objets) == {"balle", "crayon"}
    assert cfg.robot["type"] == "so101_follower"
    assert set(cfg.robot["cameras"]) == {"front", "wrist"}
    assert cfg.entrainement.politique == "act"
    assert cfg.voix.langue == "fr"


def test_noms_derives(cfg, tmp_path: Path):
    assert cfg.repo_dataset("balle") == "local/ailogos_attrape_balle"
    assert cfg.racine_dataset("balle") == tmp_path / "hf_lerobot" / "local" / "ailogos_attrape_balle"
    assert cfg.nom_entrainement("crayon") == "ailogos_act_crayon"
    sortie = tmp_path / "outputs" / "train" / "ailogos_act_balle"
    assert cfg.dossier_entrainement("balle") == sortie
    assert cfg.chemin_politique("balle") == str(sortie / "checkpoints" / "last" / "pretrained_model")
    assert not cfg.politique_disponible("balle")
    assert cfg.journal_essais == tmp_path / "outputs" / "ailogos" / "journal.jsonl"


def test_politique_imposee(tmp_path: Path):
    objets = {
        "balle": {"politique": "mon_compte/act_balle"},  # identifiant Hub : supposé disponible
        "crayon": {"politique": str(tmp_path / "absente")},  # chemin local absent
    }
    cfg = charger_configuration(ecrire_config(tmp_path, objets=objets))
    assert cfg.chemin_politique("balle") == "mon_compte/act_balle"
    assert cfg.politique_disponible("balle")
    assert not cfg.politique_disponible("crayon")
    # Valeurs par défaut d'un objet décrit au minimum.
    assert cfg.objet("balle").synonymes == ("balle",)
    assert "balle" in cfg.objet("balle").tache


def test_objet_inconnu(cfg):
    with pytest.raises(ErreurConfiguration, match="Objets connus : balle, crayon"):
        cfg.objet("tasse")


def test_arguments_cli(cfg):
    arguments = cfg.arguments_robot()
    assert "--robot.type=so101_follower" in arguments
    assert "--robot.port=/dev/ttyACM0" in arguments
    cameras = next(a for a in arguments if a.startswith("--robot.cameras="))
    assert '"front"' in cameras and '"index_or_path": 1' in cameras
    attendus = {"--teleop.type=so101_leader", "--teleop.port=/dev/ttyACM1", "--teleop.id=leader"}
    assert set(cfg.arguments_teleop()) == attendus


def test_config_robot_lerobot(cfg):
    """La section robot se décode en vraie ``RobotConfig`` LeRobot."""
    pytest.importorskip("draccus")
    robot = cfg.config_robot()
    # SO-100 et SO-101 partagent la même classe de configuration dans LeRobot.
    assert type(robot).__name__ == "SOFollowerRobotConfig"
    assert robot.port == "/dev/ttyACM0" and robot.id == "suiveur"
    assert robot.cameras["wrist"].width == 128
    teleop = cfg.config_teleop()
    assert type(teleop).__name__ == "SOLeaderTeleopConfig" and teleop.port == "/dev/ttyACM1"


@pytest.mark.parametrize(
    ("modifications", "message"),
    [
        ({"objets": {}}, "au moins un objet"),
        ({"objets": {"Balle Rouge": {}}}, "Nom d'objet invalide"),
        ({"objets": {"balle": {"couleur": "rouge"}}}, "couleur"),
        ({"robot": {"port": "/dev/ttyACM0"}}, "robot"),
        ({"voix": {"modele": "small", "inconnu": 1}}, "inconnu"),
        ({"entrainement": {"validation": 1.5}}, "validation"),
        ({"execution": {"interpolation": 0}}, "interpolation"),
        ({"envoyer_sur_hub": True}, "utilisateur_hf"),
        ({"surprise": 1}, "surprise"),
    ],
)
def test_erreurs_de_configuration(tmp_path: Path, modifications, message):
    with pytest.raises(ErreurConfiguration, match=message):
        charger_configuration(ecrire_config(tmp_path, **modifications))


def test_fichier_absent(tmp_path: Path):
    with pytest.raises(ErreurConfiguration, match="introuvable"):
        charger_configuration(tmp_path / "absent.yaml")
