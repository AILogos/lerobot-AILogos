"""Journal des essais et des commandes (un événement JSON par ligne).

``tester.py`` et ``v2_commande_vocale.py`` y écrivent ; ``monitorer.py bilan`` le lit
pour afficher le taux de réussite par objet et l'historique.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def ecrire(chemin: Path, type_evenement: str, **champs: Any) -> dict[str, Any]:
    """Ajoute un événement horodaté au journal et le retourne."""
    evenement = {"date": datetime.now().isoformat(timespec="seconds"), "type": type_evenement, **champs}
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "a", encoding="utf-8") as fichier:
        fichier.write(json.dumps(evenement, ensure_ascii=False) + "\n")
    return evenement


def lire(chemin: Path) -> list[dict[str, Any]]:
    """Lit tous les événements ; les lignes illisibles (coupure en cours d'écriture) sont ignorées."""
    if not chemin.exists():
        return []
    evenements = []
    with open(chemin, encoding="utf-8") as fichier:
        for ligne in fichier:
            try:
                evenements.append(json.loads(ligne))
            except json.JSONDecodeError:
                continue
    return evenements


@dataclass
class BilanObjet:
    objet: str
    essais: int = 0
    reussites: int = 0
    duree_totale_s: float = 0.0
    dernier: str = ""

    @property
    def taux(self) -> float | None:
        return self.reussites / self.essais if self.essais else None

    @property
    def duree_moyenne_s(self) -> float | None:
        return self.duree_totale_s / self.essais if self.essais else None


def bilan(evenements: list[dict[str, Any]]) -> dict[str, BilanObjet]:
    """Taux de réussite par objet, sur les essais notés (``succes`` vrai ou faux)."""
    resultat: dict[str, BilanObjet] = defaultdict(lambda: BilanObjet(objet=""))
    for evenement in evenements:
        if evenement.get("type") != "essai" or evenement.get("succes") is None:
            continue
        objet = str(evenement.get("objet"))
        ligne = resultat[objet]
        ligne.objet = objet
        ligne.essais += 1
        ligne.reussites += bool(evenement["succes"])
        ligne.duree_totale_s += float(evenement.get("duree_s") or 0.0)
        ligne.dernier = str(evenement.get("date", ""))
    return dict(resultat)
