"""Assistant vocal : des phrases entendues aux saisies du robot (moteur factice)."""

from __future__ import annotations

import threading

import pytest

from AILogos import journal, tester, v2_commande_vocale
from AILogos.moteur import ErreurExecution, ResultatExecution
from AILogos.v2_commande_vocale import AssistantVocal
from AILogos.voix.interpreteur import Interpreteur, TypeIntention


class MoteurFactice:
    """Enregistre les saisies demandées ; peut attendre un « stop » ou refuser un objet."""

    simulation = True

    def __init__(self, attendre_arret: bool = False, sans_politique: tuple[str, ...] = ()):
        self.saisies: list[str] = []
        self.attendre_arret = attendre_arret
        self.sans_politique = sans_politique
        self.en_cours = threading.Event()

    def politique_prete(self, objet: str) -> bool:
        return objet not in self.sans_politique

    def verifier(self, objet: str) -> None:
        if objet in self.sans_politique:
            raise ErreurExecution(f"Aucune politique entraînée pour « {objet} ».")

    def executer(self, objet: str, duree_s: float, arret: threading.Event) -> ResultatExecution:
        self.saisies.append(objet)
        self.en_cours.set()
        interrompu = arret.wait(5.0) if self.attendre_arret else arret.is_set()
        return ResultatExecution(objet, 0.1, interrompu=interrompu, simule=True)

    def prechauffer(self) -> None:
        pass

    def fermer(self) -> None:
        pass


class ParoleFactice:
    def __init__(self) -> None:
        self.phrases: list[str] = []

    def dire(self, texte: str) -> None:
        self.phrases.append(texte)


@pytest.fixture
def parole() -> ParoleFactice:
    return ParoleFactice()


def _assistant(cfg, moteur, parole) -> AssistantVocal:
    return AssistantVocal(cfg, moteur, Interpreteur.depuis_configuration(cfg), parole)


def test_attrape_la_balle(cfg, parole):
    moteur = MoteurFactice()
    assistant = _assistant(cfg, moteur, parole)
    assert assistant.recevoir("Attrape la balle !").type is TypeIntention.ATTRAPER
    assistant.traiter_file()

    assert moteur.saisies == ["balle"]
    assert parole.phrases == ["Compris : attrape la balle.", "Terminé."]
    (evenement,) = journal.lire(cfg.journal_essais)
    assert evenement["objet"] == "balle" and evenement["source"] == "voix"
    assert evenement["texte"] == "Attrape la balle !" and evenement["succes"] is None


def test_plusieurs_objets_dans_l_ordre(cfg, parole):
    moteur = MoteurFactice()
    assistant = _assistant(cfg, moteur, parole)
    assistant.recevoir("prends le stylo puis le ballon")
    assistant.traiter_file()
    assert moteur.saisies == ["crayon", "balle"]


def test_phrases_sans_effet(cfg, parole):
    moteur = MoteurFactice()
    assistant = _assistant(cfg, moteur, parole)
    assistant.recevoir("bonjour")
    assistant.recevoir("stop")  # rien en cours : sans effet
    assistant.recevoir("attrape la tasse")
    assistant.traiter_file()
    assert moteur.saisies == []
    assert parole.phrases == ["Je ne connais pas cet objet. Je sais attraper : balle, crayon."]


def test_objet_sans_politique(cfg, parole):
    moteur = MoteurFactice(sans_politique=("crayon",))
    assistant = _assistant(cfg, moteur, parole)
    assistant.recevoir("attrape le crayon puis la balle")
    assistant.traiter_file()
    # La séquence est abandonnée au premier objet impossible.
    assert moteur.saisies == []
    assert parole.phrases == ["Je n'ai pas encore appris à attraper : crayon."]


def test_stop_interrompt_la_saisie_et_la_suite(cfg, parole):
    moteur = MoteurFactice(attendre_arret=True)
    assistant = _assistant(cfg, moteur, parole)
    assistant.recevoir("attrape la balle puis le crayon")
    fil = threading.Thread(target=assistant.traiter_file, daemon=True)
    fil.start()
    assert moteur.en_cours.wait(5.0)

    # Pendant la saisie, un nouvel ordre est refusé mais « stop » passe.
    assistant.recevoir("attrape le crayon")
    assistant.recevoir("Arrête !")
    fil.join(timeout=5.0)
    assert not fil.is_alive()
    assert moteur.saisies == ["balle"]  # le crayon de la séquence est abandonné
    assert parole.phrases[-1] == "Arrêté."
    assert journal.lire(cfg.journal_essais)[0]["interrompu"] is True

    # L'assistant est de nouveau disponible.
    moteur.attendre_arret = False
    assistant.recevoir("attrape le crayon")
    assistant.traiter_file()
    assert moteur.saisies == ["balle", "crayon"]


def test_au_revoir_termine_la_boucle(cfg, parole):
    assistant = _assistant(cfg, MoteurFactice(), parole)
    fil = threading.Thread(target=assistant.boucle, daemon=True)
    fil.start()
    assistant.recevoir("au revoir")
    fil.join(timeout=5.0)
    assert not fil.is_alive()


def test_programme_v2_en_simulation(cfg, capsys, monkeypatch):
    """Le programme complet, en un ordre : texte → intention → saisie fictive → journal."""
    monkeypatch.setattr("AILogos.moteur.DUREE_FICTIVE_S", 0.05)
    arguments = ["--config", str(cfg.chemin), "--simulation", "--sans-retour-vocal"]
    assert v2_commande_vocale.principal([*arguments, "--texte", "Attrape la balle puis le crayon"]) == 0
    sortie = capsys.readouterr().out
    assert "Compris : attrape la balle." in sortie and "Compris : attrape le crayon." in sortie
    assert [e["objet"] for e in journal.lire(cfg.journal_essais)] == ["balle", "crayon"]


def test_programme_tester_en_simulation(cfg, capsys, monkeypatch):
    monkeypatch.setattr("AILogos.moteur.DUREE_FICTIVE_S", 0.05)
    arguments = [
        "--config",
        str(cfg.chemin),
        "--objet",
        "balle",
        "--simulation",
        "--sans-note",
        "--essais",
        "2",
    ]
    assert tester.principal(arguments) == 0
    essais = journal.lire(cfg.journal_essais)
    assert len(essais) == 2 and all(e["source"] == "test" and e["simulation"] for e in essais)
    # Sans politique entraînée, le test sur le vrai robot est refusé avec la marche à suivre.
    assert tester.principal(["--config", str(cfg.chemin), "--objet", "balle"]) == 1
    assert "AILogos.entrainer --objet balle" in capsys.readouterr().out


def test_tester_avec_notes(cfg, monkeypatch):
    """Campagne notée : réussi, raté avec commentaire, puis arrêt demandé par l'opérateur."""
    reponses = iter(["", "o", "", "n", "pince à côté", "q"])
    monkeypatch.setattr("builtins.input", lambda _question="": next(reponses))
    notes = tester.tester_objet(cfg, MoteurFactice(), "balle", essais=5, duree_s=1)
    assert notes == [True, False]
    essais = journal.lire(cfg.journal_essais)
    assert [e["succes"] for e in essais] == [True, False]
    assert essais[1]["commentaire"] == "pince à côté"
