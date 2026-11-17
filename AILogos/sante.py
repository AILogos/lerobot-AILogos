"""Télémétrie des moteurs (température, tension, charge, courant) et seuils d'alerte.

Les servomoteurs Feetech STS3215 du SO-101 exposent ces mesures sur le bus série.
``lire_sante`` les relève et les compare aux seuils de la section ``sante`` de
``config.yaml`` ; ``monitorer.py robot`` les affiche en continu, et ``moteur.py``
fait un relevé avant et après chaque saisie.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .configuration import Sante

OK, ALERTE, CRITIQUE = "ok", "alerte", "critique"
_ORDRE = {OK: 0, ALERTE: 1, CRITIQUE: 2}

# Registre → (clé de mesure, facteur vers l'unité affichée). Unités Feetech :
# tension en 0,1 V, charge en 0,1 % du couple maximal, courant en pas de 6,5 mA.
_REGISTRES = {
    "Present_Temperature": ("temperature_c", 1.0),
    "Present_Voltage": ("tension_v", 0.1),
    "Present_Load": ("charge_pct", 0.1),
    "Present_Current": ("courant_ma", 6.5),
    "Status": ("statut", 1.0),
}
_CHARGE_MAX_BRUTE = 0x3FF  # amplitude maximale du registre de charge (10 bits)
# Bits du registre d'état des servos Feetech STS.
_BITS_STATUT = {0: "tension", 1: "capteur d'angle", 2: "surchauffe", 3: "surintensité", 5: "surcharge"}


@dataclass
class EtatMoteur:
    nom: str
    position: float | None = None
    temperature_c: float | None = None
    tension_v: float | None = None
    charge_pct: float | None = None
    courant_ma: float | None = None
    statut: int | None = None
    niveau: str = OK
    messages: list[str] = field(default_factory=list)


@dataclass
class EtatRobot:
    horodatage: float
    moteurs: list[EtatMoteur]

    @property
    def niveau(self) -> str:
        return max((m.niveau for m in self.moteurs), key=_ORDRE.__getitem__, default=OK)

    @property
    def messages(self) -> list[str]:
        return [f"{m.nom} : {message}" for m in self.moteurs for message in m.messages]

    @property
    def temperature_max_c(self) -> float | None:
        temperatures = [m.temperature_c for m in self.moteurs if m.temperature_c is not None]
        return max(temperatures) if temperatures else None

    def en_dict(self) -> dict[str, Any]:
        return {
            "niveau": self.niveau,
            "temperature_max_c": self.temperature_max_c,
            "messages": self.messages,
            "moteurs": [asdict(m) for m in self.moteurs],
        }


def evaluer(moteur: EtatMoteur, seuils: Sante) -> EtatMoteur:
    """Renseigne ``niveau`` et ``messages`` d'un moteur à partir de ses mesures."""
    niveau, messages = OK, []

    def signaler(nouveau: str, message: str) -> None:
        nonlocal niveau
        messages.append(message)
        if _ORDRE[nouveau] > _ORDRE[niveau]:
            niveau = nouveau

    if moteur.temperature_c is not None:
        if moteur.temperature_c >= seuils.temperature_critique_c:
            signaler(CRITIQUE, f"température critique ({moteur.temperature_c:.0f} °C)")
        elif moteur.temperature_c >= seuils.temperature_alerte_c:
            signaler(ALERTE, f"température élevée ({moteur.temperature_c:.0f} °C)")
    if moteur.tension_v is not None:
        if moteur.tension_v < seuils.tension_min_v:
            signaler(ALERTE, f"tension basse ({moteur.tension_v:.1f} V) : vérifier l'alimentation")
        elif moteur.tension_v > seuils.tension_max_v:
            signaler(CRITIQUE, f"tension trop haute ({moteur.tension_v:.1f} V) : mauvaise alimentation ?")
    if moteur.charge_pct is not None and abs(moteur.charge_pct) >= seuils.charge_alerte_pct:
        signaler(ALERTE, f"charge élevée ({abs(moteur.charge_pct):.0f} %) : le bras force")
    if moteur.statut:
        defauts = [nom for bit, nom in _BITS_STATUT.items() if moteur.statut & (1 << bit)]
        if defauts:
            signaler(CRITIQUE, "défaut signalé par le moteur : " + ", ".join(defauts))

    moteur.niveau, moteur.messages = niveau, messages
    return moteur


def lire_sante(robot: Any, seuils: Sante) -> EtatRobot | None:
    """Relève l'état de tous les moteurs du robot.

    Retourne ``None`` si le robot n'a pas de bus de moteurs interrogeable (robot simulé).
    À n'appeler que lorsque aucune boucle de contrôle n'utilise le bus au même moment.
    """
    bus = getattr(robot, "bus", None)
    if bus is None or not getattr(bus, "is_connected", False):
        return None

    moteurs = {nom: EtatMoteur(nom=nom) for nom in bus.motors}
    try:
        for nom, position in bus.sync_read("Present_Position").items():
            moteurs[nom].position = float(position)
    except Exception:  # noqa: BLE001  # nosec B110 — une mesure manquante ne doit pas bloquer les autres
        pass
    for registre, (cle, facteur) in _REGISTRES.items():
        try:
            valeurs = bus.sync_read(registre, normalize=False)
        except Exception:  # noqa: BLE001, S112  # nosec B112 — registre absent sur ce modèle de moteur
            continue
        for nom, valeur in valeurs.items():
            if cle == "charge_pct" and valeur > _CHARGE_MAX_BRUTE:
                # Charge négative encore codée « signe + amplitude » (bit 10 = signe).
                valeur = -(int(valeur) & _CHARGE_MAX_BRUTE)
            setattr(moteurs[nom], cle, int(valeur) if cle == "statut" else round(float(valeur) * facteur, 2))

    return EtatRobot(horodatage=time.time(), moteurs=[evaluer(m, seuils) for m in moteurs.values()])
