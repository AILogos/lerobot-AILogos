"""Commandes ``lerobot-record`` / ``lerobot-train`` construites depuis la configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from AILogos import enregistrer, entrainer
from AILogos.commandes import (
    afficher_commande,
    commande_enregistrement,
    commande_entrainement,
    images_du_dataset,
    lancer,
    nombre_de_pas,
)
from AILogos.configuration import charger_configuration

from .conftest import ecrire_config


def _creer_dataset(cfg, objet: str, images: int, episodes: int = 5) -> None:
    meta = cfg.racine_dataset(objet) / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_frames": images, "total_episodes": episodes}))


def test_commande_enregistrement(cfg):
    commande = commande_enregistrement(cfg, "balle", episodes=12, afficher=False)
    assert commande[:3] == [sys.executable, "-m", "lerobot.scripts.lerobot_record"]
    for attendu in (
        "--robot.type=so101_follower",
        "--teleop.type=so101_leader",
        "--dataset.repo_id=local/ailogos_attrape_balle",
        "--dataset.single_task=Attrape la balle",
        "--dataset.num_episodes=12",
        "--dataset.push_to_hub=false",
        "--dataset.no_stamp=true",
        "--display_data=false",
    ):
        assert attendu in commande
    assert "--resume=true" not in commande
    assert "--resume=true" in commande_enregistrement(cfg, "balle", reprendre=True)


def test_commande_enregistrement_acceptee_par_lerobot_record(cfg):
    """La ligne de commande est décodée par le vrai analyseur de ``lerobot-record``."""
    draccus = pytest.importorskip("draccus")
    pytest.importorskip("datasets", reason="lerobot-record demande l'extra dataset")
    from lerobot.scripts.lerobot_record import RecordConfig

    commande = commande_enregistrement(cfg, "crayon", reprendre=True)
    config = draccus.parse(config_class=RecordConfig, args=commande[3:])
    # SO-100 et SO-101 partagent la même classe de configuration dans LeRobot.
    assert type(config.robot).__name__ == "SOFollowerRobotConfig"
    assert config.robot.port == "/dev/ttyACM0" and config.robot.id == "suiveur"
    assert list(config.robot.cameras) == ["front", "wrist"]
    assert config.robot.cameras["front"].index_or_path == 1
    assert type(config.teleop).__name__ == "SOLeaderTeleopConfig"
    assert config.dataset.repo_id == "local/ailogos_attrape_crayon"
    assert config.dataset.single_task == "Attrape le crayon"
    assert config.dataset.no_stamp and config.resume and not config.dataset.push_to_hub


def test_nombre_de_pas(cfg, tmp_path: Path):
    # Dataset absent : plancher.
    assert nombre_de_pas(cfg, "balle")[0] == 20_000
    # 10 époques × ceil(40 000 / 8) = 50 000 pas.
    _creer_dataset(cfg, "balle", images=40_000)
    assert images_du_dataset(cfg, "balle") == 40_000
    assert nombre_de_pas(cfg, "balle")[0] == 50_000
    # Petit dataset : le plancher s'applique.
    _creer_dataset(cfg, "crayon", images=800)
    assert nombre_de_pas(cfg, "crayon")[0] == 20_000
    # Valeur imposée dans la configuration.
    fixe = charger_configuration(ecrire_config(tmp_path, entrainement={"steps": 1234}))
    assert nombre_de_pas(fixe, "balle")[0] == 1234


def test_commande_entrainement(cfg, tmp_path: Path):
    _creer_dataset(cfg, "balle", images=40_000)
    commande = commande_entrainement(cfg, "balle", supplementaires=["--num_workers=2"])
    sortie = tmp_path / "outputs" / "train" / "ailogos_act_balle"
    assert commande[:3] == [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    for attendu in (
        "--dataset.repo_id=local/ailogos_attrape_balle",
        "--policy.type=act",
        "--policy.device=cuda",
        f"--output_dir={sortie}",
        "--job_name=ailogos_act_balle",
        "--steps=50000",
        "--save_freq=10000",
        "--policy.push_to_hub=false",
        "--dataset.eval_split=0.1",
        "--eval_steps=10000",
    ):
        assert attendu in commande
    assert commande[-1] == "--num_workers=2"
    assert not any(a.startswith("--trackio") or a.startswith("--policy.repo_id") for a in commande)

    court = commande_entrainement(cfg, "balle", steps=500, device="cpu")
    assert "--steps=500" in court and "--save_freq=500" in court and "--policy.device=cpu" in court


def test_commande_entrainement_hub_et_trackio(tmp_path: Path):
    cfg = charger_configuration(
        ecrire_config(
            tmp_path,
            utilisateur_hf="alice",
            envoyer_sur_hub=True,
            entrainement={"trackio": True, "validation": 0, "steps": 100},
        )
    )
    commande = commande_entrainement(cfg, "balle")
    assert "--dataset.repo_id=alice/ailogos_attrape_balle" in commande
    assert "--policy.push_to_hub=true" in commande
    assert "--policy.repo_id=alice/ailogos_act_balle" in commande
    assert "--trackio.enable=true" in commande
    assert not any(a.startswith("--dataset.eval_split") or a.startswith("--eval_steps") for a in commande)


def test_commande_reprise(cfg, tmp_path: Path):
    commande = commande_entrainement(cfg, "balle", reprendre=True)
    config = tmp_path / "outputs/train/ailogos_act_balle/checkpoints/last/pretrained_model/train_config.json"
    assert commande[3:] == [f"--config_path={config}", "--resume=true"]


def test_afficher_commande():
    texte = afficher_commande(["python", "-m", "module", "--a=1", "--tache=Attrape la balle"])
    assert texte == "python -m module \\\n  --a=1 \\\n  '--tache=Attrape la balle'"


def test_lancer_recopie_la_sortie_dans_le_journal(tmp_path: Path, capfdbinary):
    journal = tmp_path / "journaux" / "sortie.log"
    script = "import sys; sys.stdout.write('10%\\r50%\\r100% step:10 loss:0.5\\nfin\\n'); sys.exit(3)"
    code = lancer([sys.executable, "-c", script], journal=journal)
    assert code == 3
    lignes = journal.read_text().splitlines()
    assert lignes[1].startswith("=====")
    # Des rafraîchissements de la barre de progression, seul le dernier état est gardé.
    assert lignes[2:] == ["100% step:10 loss:0.5", "fin"]


def test_programmes_en_dry_run(cfg, capsys):
    assert enregistrer.principal(["--config", str(cfg.chemin), "--objet", "balle", "--dry-run"]) == 0
    assert "lerobot.scripts.lerobot_record" in capsys.readouterr().out
    arguments = ["--config", str(cfg.chemin), "--tous", "--dry-run", "--", "--job.target=a10g-small"]
    assert entrainer.principal(arguments) == 0
    sortie = capsys.readouterr().out
    assert sortie.count("lerobot.scripts.lerobot_train") == 2 and "--job.target=a10g-small" in sortie


def test_garde_fous_des_programmes(cfg, capsys):
    """Sans dataset on n'entraîne pas ; un dataset existant n'est pas écrasé."""
    config = ["--config", str(cfg.chemin)]
    assert entrainer.principal([*config, "--objet", "balle"]) == 1
    assert "AILogos.enregistrer --objet balle" in capsys.readouterr().out
    assert entrainer.principal([*config, "--objet", "balle", "--reprendre"]) == 1
    assert enregistrer.principal([*config, "--objet", "balle", "--reprendre"]) == 1

    _creer_dataset(cfg, "balle", images=100)
    assert enregistrer.principal([*config, "--objet", "balle"]) == 1
    assert "--reprendre" in capsys.readouterr().out
