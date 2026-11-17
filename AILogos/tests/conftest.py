"""Fixtures communes aux tests AILogos (aucun matériel requis)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from AILogos.configuration import Configuration, charger_configuration

CAMERAS = {
    "front": {"type": "opencv", "index_or_path": 1, "width": 128, "height": 96, "fps": 30},
    "wrist": {"type": "opencv", "index_or_path": 2, "width": 128, "height": 96, "fps": 30},
}


def ecrire_config(dossier: Path, **modifications: object) -> Path:
    """Écrit un config.yaml minimal dans ``dossier`` et retourne son chemin."""
    contenu: dict[str, object] = {
        "utilisateur_hf": "local",
        "dossier_sorties": str(dossier / "outputs"),
        "robot": {"type": "so101_follower", "port": "/dev/ttyACM0", "id": "suiveur", "cameras": CAMERAS},
        "teleop": {"type": "so101_leader", "port": "/dev/ttyACM1", "id": "leader"},
        "objets": {
            "balle": {"tache": "Attrape la balle", "synonymes": ["balle", "ballon"]},
            "crayon": {"tache": "Attrape le crayon", "synonymes": ["crayon", "stylo"]},
        },
        "execution": {"duree_s": 1, "device": "cpu"},
    }
    contenu.update(modifications)
    chemin = dossier / "config.yaml"
    chemin.write_text(yaml.safe_dump(contenu, allow_unicode=True), encoding="utf-8")
    return chemin


@pytest.fixture
def cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Configuration:
    # Les datasets sont cherchés sous HF_LEROBOT_HOME : on l'isole du vrai cache.
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "hf_lerobot"))
    return charger_configuration(ecrire_config(tmp_path))


@pytest.fixture(scope="session")
def checkpoint_act(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Petite politique ACT non entraînée, au format d'un checkpoint ``lerobot-train``."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("datasets", reason="lerobot.rollout demande l'extra dataset")
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTConfig, ACTPolicy

    formes = {
        "observation.state": (6,),
        "observation.images.front": (3, 96, 128),
        "observation.images.wrist": (3, 96, 128),
        "action": (6,),
    }
    config = ACTConfig(
        device="cpu",
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, formes["observation.state"]),
            "observation.images.front": PolicyFeature(FeatureType.VISUAL, formes["observation.images.front"]),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, formes["observation.images.wrist"]),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, formes["action"])},
        chunk_size=10,
        n_action_steps=10,
        dim_model=64,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,  # pas de téléchargement des poids ImageNet
        push_to_hub=False,
    )
    statistiques = {}
    for nom, forme in formes.items():
        forme_stat = (forme[0], 1, 1) if len(forme) == 3 else forme
        statistiques[nom] = {
            "mean": torch.zeros(forme_stat),
            "std": torch.ones(forme_stat),
            "min": -torch.ones(forme_stat),
            "max": torch.ones(forme_stat),
        }
    dossier = tmp_path_factory.mktemp("politique") / "pretrained_model"
    ACTPolicy(config).save_pretrained(dossier)
    preprocesseur, postprocesseur = make_pre_post_processors(config, dataset_stats=statistiques)
    preprocesseur.save_pretrained(dossier)
    postprocesseur.save_pretrained(dossier)
    return dossier
