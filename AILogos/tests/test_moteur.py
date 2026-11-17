"""Moteur d'exécution : la vraie boucle ``lerobot.rollout`` sur le robot simulé."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from AILogos.configuration import charger_configuration
from AILogos.moteur import ErreurExecution, MoteurSaisie

from .conftest import ecrire_config


def test_simulation_sans_politique_est_fictive(cfg, monkeypatch):
    monkeypatch.setattr("AILogos.moteur.DUREE_FICTIVE_S", 0.05)
    resultat = MoteurSaisie(cfg, simulation=True).executer("balle")
    assert resultat.ok and resultat.simule and not resultat.interrompu


def test_simulation_fictive_s_arrete_sur_demande(cfg):
    arret = threading.Event()
    arret.set()
    resultat = MoteurSaisie(cfg, simulation=True).executer("balle", duree_s=30, arret=arret)
    assert resultat.interrompu and resultat.duree_s < 1


def test_robot_reel_refuse_sans_politique(cfg):
    with pytest.raises(ErreurExecution, match="AILogos.entrainer --objet balle"):
        MoteurSaisie(cfg).executer("balle")


def test_objet_inconnu(cfg):
    with pytest.raises(ValueError, match="Objet inconnu"):
        MoteurSaisie(cfg, simulation=True).executer("tasse")


@pytest.fixture
def cfg_avec_politique(tmp_path: Path, checkpoint_act: Path):
    objets = {"balle": {"tache": "Attrape la balle", "politique": str(checkpoint_act)}}
    return charger_configuration(ecrire_config(tmp_path, objets=objets))


def test_politique_sur_robot_simule(cfg_avec_politique):
    """La politique est chargée et pilote le robot simulé jusqu'à la durée demandée."""
    moteur = MoteurSaisie(cfg_avec_politique, simulation=True)
    assert moteur.politique_prete("balle")
    resultat = moteur.executer("balle", duree_s=0.5)
    assert resultat.ok, resultat.erreur
    assert resultat.simule and not resultat.interrompu
    assert 0.4 <= resultat.duree_s < 3


def test_arret_pendant_la_saisie(cfg_avec_politique):
    """Un « stop » venu d'un autre fil interrompt la saisie bien avant la durée maximale."""
    moteur = MoteurSaisie(cfg_avec_politique, simulation=True)
    arret = threading.Event()
    resultat = moteur.executer("balle", duree_s=0.2)  # première exécution : charge torch et la politique
    assert resultat.ok, resultat.erreur

    minuteur = threading.Timer(0.5, arret.set)
    minuteur.start()
    debut = time.perf_counter()
    resultat = moteur.executer("balle", duree_s=60, arret=arret)
    assert resultat.ok, resultat.erreur
    assert resultat.interrompu
    assert time.perf_counter() - debut < 30


def test_materiel_absent_donne_une_erreur_lisible(cfg_avec_politique):
    """Port série inexistant : une ``ErreurExecution`` claire, pas une trace Python."""
    moteur = MoteurSaisie(cfg_avec_politique)  # vrai robot demandé, mais rien n'est branché
    with pytest.raises(ErreurExecution, match="Impossible de préparer la saisie de « balle »"):
        moteur.executer("balle", duree_s=0.5)


def test_panne_pendant_la_saisie(cfg_avec_politique, monkeypatch):
    """Une panne du robot en cours de saisie est rapportée dans le résultat."""
    from AILogos.robot_simule import RobotSimule

    def panne(self, action):
        raise ConnectionError("moteur muet")

    moteur = MoteurSaisie(cfg_avec_politique, simulation=True)
    monkeypatch.setattr(RobotSimule, "send_action", panne)
    resultat = moteur.executer("balle", duree_s=0.5)
    assert not resultat.ok and "moteur muet" in resultat.erreur
