"""Moteur d'exécution : fait attraper un objet au robot avec la politique entraînée.

``MoteurSaisie.executer("balle")`` est la brique commune à ``tester.py`` et à la
commande vocale. Il s'appuie sur le moteur officiel de LeRobot (``lerobot.rollout``,
celui de ``lerobot-rollout``) : connexion du bras, boucle de contrôle cadencée,
puis retour en douceur à la position de départ et coupure du couple.

Une saisie = une connexion : entre deux commandes, le bras est au repos, moteurs relâchés.
Le bras doit donc être en position de repos au lancement de chaque commande.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from .configuration import Configuration
from .robot_simule import section_robot_simule
from .sante import CRITIQUE, lire_sante

logger = logging.getLogger(__name__)

DUREE_FICTIVE_S = 3.0


class ErreurExecution(RuntimeError):
    """La saisie n'a pas pu être lancée (politique absente, moteur en défaut…)."""


@dataclass
class ResultatExecution:
    objet: str
    duree_s: float
    interrompu: bool = False  # arrêt demandé (voix, Ctrl+C) avant la durée prévue
    simule: bool = False
    erreur: str | None = None
    sante_avant: dict[str, Any] | None = None
    sante_apres: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.erreur is None


class MoteurSaisie:
    """Exécute la politique associée à un objet sur le robot décrit dans ``config.yaml``.

    Avec ``simulation=True``, aucun matériel n'est utilisé : la politique tourne sur un
    robot simulé si elle est entraînée, sinon l'exécution est fictive (une simple attente).
    """

    def __init__(self, cfg: Configuration, *, simulation: bool = False, afficher: bool = False):
        self.simulation = simulation
        self.afficher = afficher
        self._visualisation_ouverte = False
        if simulation:
            cfg = dataclasses.replace(cfg, robot=section_robot_simule(cfg.robot))
        self.cfg = cfg

    # ── Vérifications ────────────────────────────────────────────────────────

    def politique_prete(self, objet: str) -> bool:
        return self.cfg.politique_disponible(objet)

    def verifier(self, objet: str) -> None:
        """Lève ``ErreurExecution`` si l'objet ne peut pas être attrapé en l'état."""
        self.cfg.objet(objet)
        if not self.simulation and not self.politique_prete(objet):
            raise ErreurExecution(
                f"Aucune politique entraînée pour « {objet} » ({self.cfg.chemin_politique(objet)}).\n"
                f"Lancez d'abord : uv run python -m AILogos.entrainer --objet {objet}"
            )

    def prechauffer(self) -> None:
        """Charge torch et le moteur de rollout à l'avance (quelques secondes, une seule fois)."""
        if any(self.politique_prete(objet) for objet in self.cfg.objets):
            import lerobot.rollout  # noqa: F401

    # ── Exécution ────────────────────────────────────────────────────────────

    def executer(
        self,
        objet: str,
        duree_s: float | None = None,
        arret: threading.Event | None = None,
    ) -> ResultatExecution:
        """Fait attraper ``objet`` pendant au plus ``duree_s`` secondes.

        ``arret`` est un événement que n'importe quel fil peut déclencher pour arrêter la
        saisie (commande vocale « stop ») ; Ctrl+C a le même effet. Dans tous les cas le
        bras revient à sa position de départ avant d'être relâché.
        """
        self.verifier(objet)
        duree_s = float(duree_s if duree_s is not None else self.cfg.execution.duree_s)
        arret = arret or threading.Event()
        if self.simulation and not self.politique_prete(objet):
            return self._executer_fictif(objet, duree_s, arret)
        return self._executer_politique(objet, duree_s, arret)

    def _executer_fictif(self, objet: str, duree_s: float, arret: threading.Event) -> ResultatExecution:
        debut = time.perf_counter()
        interrompu = False
        try:
            interrompu = arret.wait(min(duree_s, DUREE_FICTIVE_S))
        except KeyboardInterrupt:
            interrompu = True
        return ResultatExecution(objet, time.perf_counter() - debut, interrompu=interrompu, simule=True)

    def _executer_politique(self, objet: str, duree_s: float, arret: threading.Event) -> ResultatExecution:
        # Imports tardifs : torch et lerobot ne sont chargés qu'au premier vrai mouvement.
        from lerobot.configs import PreTrainedConfig
        from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context, create_strategy

        cfg = self.cfg
        chemin = cfg.chemin_politique(objet)
        resultat = ResultatExecution(objet, 0.0, simule=self.simulation)
        try:
            config_politique = PreTrainedConfig.from_pretrained(chemin)
            config_politique.pretrained_path = chemin
            config_rollout = RolloutConfig(
                robot=cfg.config_robot(),
                policy=config_politique,
                strategy=BaseStrategyConfig(),
                fps=cfg.enregistrement.fps,
                duration=duree_s,
                task=cfg.objet(objet).tache,
                device=cfg.execution.device,
                interpolation_multiplier=cfg.execution.interpolation,
                display_data=self.afficher,
                play_sounds=False,
                return_to_initial_position=True,
            )
            self._ouvrir_visualisation()
            # Charge la politique puis connecte le robot (la politique d'abord : un mauvais
            # chemin échoue avant que le bras ne soit mis sous couple).
            contexte = build_rollout_context(config_rollout, arret)
            strategie = create_strategy(config_rollout.strategy)
        except Exception as erreur:
            # Port série absent, caméra débranchée, checkpoint illisible… : rien n'a bougé.
            raise ErreurExecution(
                f"Impossible de préparer la saisie de « {objet} » ({type(erreur).__name__}) : {erreur}"
            ) from erreur

        robot = contexte.hardware.robot_wrapper.inner
        debut = time.perf_counter()
        try:
            strategie.setup(contexte)
            avant = lire_sante(robot, cfg.sante)
            if avant is not None:
                resultat.sante_avant = avant.en_dict()
                if avant.niveau == CRITIQUE:
                    raise ErreurExecution("Saisie refusée, moteur en défaut : " + " ; ".join(avant.messages))
            debut = time.perf_counter()
            strategie.run(contexte)
            resultat.duree_s = time.perf_counter() - debut
            resultat.interrompu = arret.is_set()
            apres = lire_sante(robot, cfg.sante)
            if apres is not None:
                resultat.sante_apres = apres.en_dict()
        except KeyboardInterrupt:
            resultat.duree_s = time.perf_counter() - debut
            resultat.interrompu = True
        except ErreurExecution:
            raise
        except Exception as erreur:
            # Panne en cours de saisie (moteur muet, caméra perdue) : signalée, sans tuer le programme.
            logger.exception("Erreur pendant la saisie de « %s »", objet)
            resultat.duree_s = time.perf_counter() - debut
            resultat.erreur = f"{type(erreur).__name__} : {erreur}"
        finally:
            # Retour à la position de départ puis déconnexion (couple coupé).
            try:
                strategie.teardown(contexte)
            except Exception as erreur:
                logger.exception("Erreur pendant le retour au repos")
                resultat.erreur = resultat.erreur or f"retour au repos impossible : {erreur}"

        inference = contexte.policy.inference
        if inference.failed and resultat.erreur is None:
            resultat.erreur = f"l'inférence s'est arrêtée sur une erreur :\n{inference.failure_traceback}"
        return resultat

    # ── Visualisation Rerun (optionnelle) ────────────────────────────────────

    def _ouvrir_visualisation(self) -> None:
        if self.afficher and not self._visualisation_ouverte:
            from lerobot.utils.visualization_utils import init_visualization

            init_visualization("rerun", session_name="ailogos")
            self._visualisation_ouverte = True

    def fermer(self) -> None:
        if self._visualisation_ouverte:
            from lerobot.utils.visualization_utils import shutdown_visualization

            shutdown_visualization("rerun")
            self._visualisation_ouverte = False
