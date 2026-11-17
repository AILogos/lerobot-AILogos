"""Télémétrie des moteurs, journal des essais et vues du monitoring."""

from __future__ import annotations

import json

from AILogos import journal, monitorer
from AILogos.configuration import Sante
from AILogos.sante import ALERTE, CRITIQUE, OK, EtatMoteur, evaluer, lire_sante

SEUILS = Sante()


class BusFactice:
    """Bus de moteurs minimal : valeurs brutes dans les unités Feetech."""

    is_connected = True

    def __init__(self, registres: dict[str, dict[str, float]]):
        self.motors = {"shoulder_pan": None, "gripper": None}
        self.registres = registres

    def sync_read(self, registre: str, normalize: bool = True) -> dict[str, float]:
        if registre not in self.registres:
            raise RuntimeError(f"registre {registre} absent")
        return self.registres[registre]


class RobotFactice:
    def __init__(self, bus: BusFactice | None):
        self.bus = bus


def test_evaluer_seuils():
    assert evaluer(EtatMoteur("m", temperature_c=40, tension_v=5.1, charge_pct=20), SEUILS).niveau == OK
    chaud = evaluer(EtatMoteur("m", temperature_c=58), SEUILS)
    assert chaud.niveau == ALERTE and "58" in chaud.messages[0]
    assert evaluer(EtatMoteur("m", temperature_c=70), SEUILS).niveau == CRITIQUE
    assert evaluer(EtatMoteur("m", tension_v=4.0), SEUILS).niveau == ALERTE
    assert evaluer(EtatMoteur("m", tension_v=14.0), SEUILS).niveau == CRITIQUE
    assert evaluer(EtatMoteur("m", charge_pct=-92.0), SEUILS).niveau == ALERTE
    # Bit 5 du registre d'état : surcharge signalée par le moteur lui-même.
    defaut = evaluer(EtatMoteur("m", statut=0b100000), SEUILS)
    assert defaut.niveau == CRITIQUE and "surcharge" in defaut.messages[0]


def test_lire_sante_convertit_les_unites():
    bus = BusFactice(
        {
            "Present_Position": {"shoulder_pan": 12.5, "gripper": 40.0},
            "Present_Temperature": {"shoulder_pan": 35, "gripper": 67},
            "Present_Voltage": {"shoulder_pan": 51, "gripper": 50},  # 0,1 V
            "Present_Load": {"shoulder_pan": 120, "gripper": -900},  # 0,1 %
            "Present_Current": {"shoulder_pan": 10, "gripper": 100},  # 6,5 mA
            # « Status » absent : la lecture continue sans lui.
        }
    )
    etat = lire_sante(RobotFactice(bus), SEUILS)
    assert etat is not None
    epaule, pince = etat.moteurs
    assert (epaule.position, epaule.temperature_c, epaule.tension_v) == (12.5, 35.0, 5.1)
    assert epaule.charge_pct == 12.0 and epaule.courant_ma == 65.0 and epaule.niveau == OK
    assert pince.niveau == CRITIQUE and len(pince.messages) == 2  # surchauffe + charge élevée
    assert etat.niveau == CRITIQUE and etat.temperature_max_c == 67.0
    assert any("gripper" in message for message in etat.messages)
    json.dumps(etat.en_dict())  # sérialisable dans le journal

    # Une charge négative non décodée par le bus (bit de signe 10) est décodée ici.
    bus.registres["Present_Load"] = {"shoulder_pan": 1024 + 300, "gripper": 0}
    assert lire_sante(RobotFactice(bus), SEUILS).moteurs[0].charge_pct == -30.0
    assert "gripper" in monitorer.tableau_sante(etat)


def test_lire_sante_sans_bus():
    assert lire_sante(RobotFactice(None), SEUILS) is None
    assert lire_sante(object(), SEUILS) is None


def test_journal_et_bilan(tmp_path):
    chemin = tmp_path / "sous" / "journal.jsonl"
    assert journal.lire(chemin) == []
    journal.ecrire(chemin, "essai", objet="balle", succes=True, duree_s=6.0, source="test")
    journal.ecrire(chemin, "essai", objet="balle", succes=False, duree_s=10.0, commentaire="pince à côté")
    journal.ecrire(
        chemin, "essai", objet="balle", succes=None, duree_s=5.0, source="voix", texte="attrape la balle"
    )
    journal.ecrire(chemin, "essai", objet="crayon", succes=True, duree_s=8.0)
    with open(chemin, "a") as fichier:
        fichier.write('{"ligne": "coupée')  # écriture interrompue : ignorée à la lecture

    evenements = journal.lire(chemin)
    assert len(evenements) == 4
    bilans = journal.bilan(evenements)
    assert (bilans["balle"].essais, bilans["balle"].reussites, bilans["balle"].taux) == (2, 1, 0.5)
    assert bilans["balle"].duree_moyenne_s == 8.0
    assert bilans["crayon"].taux == 1.0

    rapport = monitorer.rapport_bilan(evenements)
    assert "50%" in rapport and "100%" in rapport
    assert "« attrape la balle »" in rapport and "pince à côté" in rapport
    assert "Aucun essai" in monitorer.rapport_bilan([])


JOURNAL_ENTRAINEMENT = """
===== python -m lerobot.scripts.lerobot_train --steps=100 =====
INFO 2026-01-01 10:00:00 ot_train.py:791 cfg.steps=100 (100)
Training:  10%|#         | 10/100 [00:02<00:25,  3.5step/s]INFO 2026-01-01 10:00:03 ot_train.py:894 step:10 smpl:80 ep:2 epch:0.59 loss:60.000 grdn:282.1 lr:1.0e-05 data_s:0.038 updt_s:0.258 step_s:0.300 smp/s:27 l1_loss:0.900 kld_loss:5.9
Traceback (most recent call last):

===== python -m lerobot.scripts.lerobot_train --steps=4000 =====
INFO 2026-01-01 11:00:00 ot_train.py:791 cfg.steps=4000 (4K)
Training:  25%|##        | 1000/4000 [05:00<15:00,  3.3step/s]INFO 2026-01-01 11:05:00 ot_train.py:894 step:1K smpl:8K ep:20 epch:1.00 loss:2.000 grdn:20.1 lr:1.0e-05 data_s:0.038 updt_s:0.258 step_s:0.300 smp/s:27 l1_loss:0.400 kld_loss:0.9
INFO 2026-01-01 11:05:01 ot_train.py:928 step 1000: eval_loss=0.5000
INFO 2026-01-01 11:05:02 ot_train.py:936 Checkpoint policy after step 1000
INFO 2026-01-01 11:10:00 ot_train.py:894 step:2K smpl:16K ep:40 epch:2.00 loss:1.000 grdn:10.1 lr:1.0e-05 data_s:0.038 updt_s:0.258 step_s:0.250 smp/s:27 l1_loss:0.200 kld_loss:0.5
INFO 2026-01-01 11:10:01 ot_train.py:928 step 2000: eval_loss=0.3000
"""


def test_analyser_journal_entrainement():
    etat = monitorer.analyser_journal(JOURNAL_ENTRAINEMENT)
    # Seul le dernier lancement compte : l'erreur du premier n'apparaît plus.
    assert etat.total == 4000 and etat.erreur is None and not etat.termine
    assert etat.pas == 2000  # « step:2K » quand la barre de progression est absente de la ligne
    assert etat.pertes == [2.0, 1.0] and etat.perte_l1 == 0.2
    assert etat.validations == [(1000, 0.5), (2000, 0.3)] and etat.checkpoints == 1
    assert etat.avancement == 0.5 and etat.restant_s == 500.0

    rapport = monitorer.rapport_entrainement("balle", etat)
    assert "50%" in rapport and "2000/4000" in rapport and "8 min 20 s" in rapport

    fini = monitorer.analyser_journal(JOURNAL_ENTRAINEMENT + "INFO ... End of training\n")
    assert fini.termine and fini.pas == 4000 and fini.restant_s is None
    assert monitorer.analyser_journal(JOURNAL_ENTRAINEMENT.split("\n=====")[1]).erreur


def test_pas_exact_depuis_la_barre_de_progression():
    ligne = (
        "Training: 31%|###| 1234/4000 [05:00<15:00]INFO x step:1K smpl:8K epch:1.00 loss:2.000 step_s:0.300\n"
    )
    assert monitorer.analyser_journal(ligne).pas == 1234


def test_courbe():
    assert monitorer.courbe([]) == ""
    assert monitorer.courbe([3.0, 2.0, 1.0]) == "█▄▁"
    assert len(monitorer.courbe(list(range(500)))) == 50


def test_vues_bilan_et_ensemble(cfg, capsys):
    journal.ecrire(cfg.journal_essais, "essai", objet="balle", succes=True, duree_s=5.0)
    assert monitorer.principal(["--config", str(cfg.chemin), "bilan", "--objet", "balle"]) == 0
    assert "100%" in capsys.readouterr().out

    assert monitorer.principal(["--config", str(cfg.chemin)]) == 0
    sortie = capsys.readouterr().out
    assert "aucune démonstration enregistrée" in sortie
    assert "AILogos.enregistrer --objet crayon" in sortie and "1/1 réussis" in sortie

    assert monitorer.principal(["--config", str(cfg.chemin), "entrainement"]) == 0
    assert "pas encore lancé" in capsys.readouterr().out
