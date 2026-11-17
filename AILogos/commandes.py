"""Construction et lancement des commandes LeRobot (``lerobot-record``, ``lerobot-train``).

Les programmes AILogos ne réécrivent ni l'enregistrement ni l'entraînement : ils
préparent la bonne ligne de commande à partir de ``config.yaml`` et la lancent.
Chaque programme accepte ``--dry-run`` pour afficher la commande sans l'exécuter.
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

from .configuration import Configuration

MODULE_ENREGISTREMENT = "lerobot.scripts.lerobot_record"
MODULE_ENTRAINEMENT = "lerobot.scripts.lerobot_train"


def _bool(valeur: bool) -> str:
    return "true" if valeur else "false"


def commande_enregistrement(
    cfg: Configuration,
    objet: str,
    *,
    episodes: int | None = None,
    reprendre: bool = False,
    afficher: bool = True,
    supplementaires: list[str] | None = None,
) -> list[str]:
    """Commande ``lerobot-record`` pour enregistrer des démonstrations de saisie d'un objet."""
    enr = cfg.enregistrement
    commande = [
        sys.executable,
        "-m",
        MODULE_ENREGISTREMENT,
        *cfg.arguments_robot(),
        *cfg.arguments_teleop(),
        f"--dataset.repo_id={cfg.repo_dataset(objet)}",
        f"--dataset.single_task={cfg.objet(objet).tache}",
        f"--dataset.num_episodes={episodes if episodes is not None else enr.episodes}",
        f"--dataset.episode_time_s={enr.duree_episode_s}",
        f"--dataset.reset_time_s={enr.duree_reset_s}",
        f"--dataset.fps={enr.fps}",
        f"--dataset.push_to_hub={_bool(cfg.envoyer_sur_hub)}",
        # Nom stable (sans date) : entrainer.py et tester.py retrouvent le dataset sans aide.
        "--dataset.no_stamp=true",
        f"--display_data={_bool(afficher)}",
    ]
    if reprendre:
        commande.append("--resume=true")
    return commande + list(supplementaires or [])


def images_du_dataset(cfg: Configuration, objet: str) -> int | None:
    """Nombre d'images du dataset local, ou ``None`` s'il n'est pas sur ce disque."""
    info = cfg.racine_dataset(objet) / "meta" / "info.json"
    try:
        return int(json.loads(info.read_text(encoding="utf-8"))["total_frames"])
    except (OSError, ValueError, KeyError):
        return None


def nombre_de_pas(cfg: Configuration, objet: str) -> tuple[int, str]:
    """Nombre de pas d'entraînement et explication de son origine.

    Règle du guide LeRobot : raisonner en époques (5 à 10 passes sur le dataset),
    puis convertir en pas : ``pas = époques × ceil(images / batch_size)``.
    """
    ent = cfg.entrainement
    if ent.steps is not None:
        return ent.steps, "fixé par « entrainement.steps »"
    images = images_du_dataset(cfg, objet)
    if images is None:
        return ent.steps_min, "dataset absent du disque : valeur de « entrainement.steps_min »"
    calcule = math.ceil(ent.epoques * math.ceil(images / ent.batch_size))
    if calcule < ent.steps_min:
        return ent.steps_min, f"{images} images : plancher « steps_min » appliqué ({calcule} calculés)"
    return calcule, f"{ent.epoques:g} époques × {images} images / batch de {ent.batch_size}"


def commande_entrainement(
    cfg: Configuration,
    objet: str,
    *,
    steps: int | None = None,
    device: str | None = None,
    reprendre: bool = False,
    supplementaires: list[str] | None = None,
) -> list[str]:
    """Commande ``lerobot-train`` pour entraîner la politique de saisie d'un objet."""
    ent = cfg.entrainement
    sortie = cfg.dossier_entrainement(objet)
    base = [sys.executable, "-m", MODULE_ENTRAINEMENT]

    if reprendre:
        config_checkpoint = sortie / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
        commande = [*base, f"--config_path={config_checkpoint}", "--resume=true"]
        if steps is not None:
            commande.append(f"--steps={steps}")
        return commande + list(supplementaires or [])

    total = steps if steps is not None else nombre_de_pas(cfg, objet)[0]
    sauvegarde = min(ent.sauvegarde_tous_les, total)
    commande = [
        *base,
        f"--dataset.repo_id={cfg.repo_dataset(objet)}",
        f"--policy.type={ent.politique}",
        f"--policy.device={device or ent.device}",
        f"--output_dir={sortie}",
        f"--job_name={cfg.nom_entrainement(objet)}",
        f"--batch_size={ent.batch_size}",
        f"--steps={total}",
        f"--save_freq={sauvegarde}",
        f"--log_freq={ent.journal_tous_les}",
        f"--policy.push_to_hub={_bool(cfg.envoyer_sur_hub)}",
    ]
    if cfg.envoyer_sur_hub:
        commande.append(f"--policy.repo_id={cfg.repo_politique(objet)}")
    if ent.validation > 0:
        commande += [f"--dataset.eval_split={ent.validation}", f"--eval_steps={sauvegarde}"]
    if ent.trackio:
        commande += ["--trackio.enable=true", "--trackio.project=ailogos"]
    return commande + list(supplementaires or [])


def afficher_commande(commande: list[str]) -> str:
    """Commande prête à copier-coller dans un terminal."""
    # « python -m module » reste sur la première ligne, puis une option par ligne.
    entete = shlex.join(commande[:3])
    return " \\\n  ".join([entete, *(shlex.quote(argument) for argument in commande[3:])])


def lancer(commande: list[str], journal: Path | None = None) -> int:
    """Lance la commande ; avec ``journal``, recopie aussi sa sortie dans ce fichier.

    Le journal est ce que ``monitorer.py entrainement`` lit pour suivre la progression.
    """
    if journal is None:
        return subprocess.run(commande, check=False).returncode  # noqa: S603  # nosec B603

    journal.parent.mkdir(parents=True, exist_ok=True)
    with (
        open(journal, "ab") as fichier,
        subprocess.Popen(commande, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as processus,  # noqa: S603  # nosec B603
    ):
        fichier.write(f"\n===== {' '.join(commande)} =====\n".encode())
        if processus.stdout is None:
            raise RuntimeError("sortie du processus indisponible")
        reste = b""
        try:
            # Lecture binaire : la barre de progression (retours chariot) s'affiche comme
            # dans un terminal, sans être découpée en une ligne par rafraîchissement.
            while morceau := processus.stdout.read1(4096):
                sys.stdout.buffer.write(morceau)
                sys.stdout.buffer.flush()
                *lignes, reste = (reste + morceau).split(b"\n")
                for ligne in lignes:
                    # Dans le journal, seul le dernier état de chaque ligne est gardé.
                    fichier.write(ligne.rsplit(b"\r", 1)[-1] + b"\n")
                fichier.flush()
        except KeyboardInterrupt:
            # Ctrl+C atteint aussi le processus fils : on le laisse s'arrêter proprement.
            processus.wait()
            raise
        finally:
            if reste:
                fichier.write(reste.rsplit(b"\r", 1)[-1] + b"\n")
        return processus.wait()
