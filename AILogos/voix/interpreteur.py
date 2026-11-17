"""Traduit une phrase (« attrape la balle ») en intention exploitable par le robot.

Module en Python pur, sans dépendance : il se teste sans micro, sans modèle et sans robot.

Règles
------
* **Arrêt** : un mot d'arrêt (« stop », « arrête »…) l'emporte sur tout le reste, et
  n'a jamais besoin du mot de réveil — on doit toujours pouvoir arrêter le bras.
* **Saisie** : il faut un verbe de saisie *et* un objet connu. Un objet prononcé seul
  dans une conversation ne déclenche donc aucun mouvement.
* **Négation** : « n'attrape pas la balle » ne déclenche rien.
* **Plusieurs objets** : « attrape la balle puis le crayon » donne les deux, dans l'ordre.
* **Tolérance** : accents, majuscules, ponctuation et petites erreurs de transcription
  (« atrape la bale ») sont absorbés.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..configuration import Configuration


class TypeIntention(StrEnum):
    ATTRAPER = "attraper"
    STOP = "stop"
    QUITTER = "quitter"
    AIDE = "aide"
    OBJET_INCONNU = "objet_inconnu"  # verbe de saisie compris, mais aucun objet connu
    AUCUNE = "aucune"


@dataclass(frozen=True)
class Intention:
    type: TypeIntention
    objets: tuple[str, ...] = ()
    texte: str = ""
    raison: str = ""


VERBES_SAISIE = frozenset(
    [
        "attrape",
        "attraper",
        "attrapes",
        "attrapez",
        "prends",
        "prend",
        "prenez",
        "prendre",
        "saisis",
        "saisit",
        "saisir",
        "saisissez",
        "ramasse",
        "ramasser",
        "ramasses",
        "ramassez",
        "recupere",
        "recuperer",
        "recuperes",
        "recuperez",
        "chope",
        "choper",
        "donne",
        "donnez",
        "passe",
        "passez",
        "apporte",
        "apportez",
        "grab",
        "take",
        "pick",
        "catch",
    ]
)
MOTS_STOP = frozenset(
    [
        "stop",
        "stoppe",
        "stopper",
        "arrete",
        "arreter",
        "arretes",
        "arretez",
        "halte",
        "annule",
        "annuler",
        "pause",
    ]
)
MOTS_QUITTER = frozenset(["quitte", "quitter", "quittez"])
PHRASES_QUITTER = ("au revoir", "fin du programme", "termine le programme", "eteins toi")
MOTS_AIDE = frozenset(["aide", "aider"])
PHRASES_AIDE = ("que sais tu", "quels objets", "que peux tu")
MOTS_NEGATION = frozenset(["pas", "jamais", "plus"])

SEUIL_VERBE = 0.85
SEUIL_OBJET = 0.75
LONGUEUR_MIN_APPROX = 4  # en dessous, seule la correspondance exacte compte


def normaliser(texte: str) -> str:
    """Minuscules, sans accents ni ponctuation : « N'attrape pas ! » → « n attrape pas »."""
    texte = unicodedata.normalize("NFD", texte.lower())
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = re.sub(r"[^a-z0-9]+", " ", texte)
    return texte.strip()


def _singulier(mot: str) -> str:
    return mot[:-1] if len(mot) > 3 and mot[-1] in "sx" else mot


def _proche(mot: str, reference: str, seuil: float) -> bool:
    if mot == reference or _singulier(mot) == reference:
        return True
    if min(len(mot), len(reference)) < LONGUEUR_MIN_APPROX:
        return False
    return difflib.SequenceMatcher(None, mot, reference).ratio() >= seuil


def _contient_phrase(mots: list[str], phrase: str) -> bool:
    cible = phrase.split()
    return any(mots[i : i + len(cible)] == cible for i in range(len(mots) - len(cible) + 1))


class Interpreteur:
    """Reconnaît les ordres destinés au robot dans du texte libre.

    Args:
        objets: nom d'objet → synonymes acceptés (le nom lui-même est toujours accepté).
        mot_reveil: s'il est renseigné, un ordre de saisie doit le contenir
            (« robot, attrape la balle ») ; tout ce qui le précède est ignoré.
    """

    def __init__(
        self,
        objets: Mapping[str, Iterable[str]],
        mot_reveil: str = "",
        exemples: Iterable[str] = (),
    ):
        self.objets: dict[str, list[list[str]]] = {}
        self._vocabulaire: list[str] = []
        for nom, synonymes in objets.items():
            synonymes = list(synonymes)
            self._vocabulaire += [s for s in synonymes if s not in self._vocabulaire]
            variantes = {normaliser(nom), *(normaliser(s) for s in synonymes)}
            # Les synonymes les plus longs d'abord : « balle de tennis » avant « balle ».
            self.objets[nom] = sorted((v.split() for v in variantes if v), key=len, reverse=True)
        self.mot_reveil = normaliser(mot_reveil)
        self.exemples = [e.strip().rstrip(".") for e in exemples if e.strip()]

    @classmethod
    def depuis_configuration(cls, cfg: Configuration) -> Interpreteur:
        return cls(
            {nom: objet.synonymes for nom, objet in cfg.objets.items()},
            cfg.voix.mot_reveil,
            exemples=[objet.tache for objet in cfg.objets.values()],
        )

    # ── Aides ────────────────────────────────────────────────────────────────

    def phrase_amorce(self) -> str:
        """Texte d'exemple donné à Whisper pour orienter la transcription vers ce vocabulaire."""
        prefixe = f"{self.mot_reveil.capitalize()}, " if self.mot_reveil else ""
        exemples = " ".join(f"{prefixe}{exemple}." for exemple in self.exemples)
        mots = ", ".join(self._vocabulaire or self.objets)
        return f"Commandes pour le robot. {exemples} Objets : {mots}. Stop. Arrête.".replace("  ", " ")

    def objets_connus(self) -> str:
        return ", ".join(self.objets)

    def _objets_mentionnes(self, mots: list[str]) -> list[str]:
        """Objets connus présents dans la phrase, dans leur ordre d'apparition."""
        trouves: list[tuple[int, str]] = []
        for nom, variantes in self.objets.items():
            position = self._position(mots, variantes)
            if position is not None:
                trouves.append((position, nom))
        return [nom for _, nom in sorted(trouves)]

    @staticmethod
    def _position(mots: list[str], variantes: list[list[str]]) -> int | None:
        for variante in variantes:
            for i in range(len(mots) - len(variante) + 1):
                if all(_proche(mots[i + j], cible, SEUIL_OBJET) for j, cible in enumerate(variante)):
                    return i
        return None

    # ── Interprétation ───────────────────────────────────────────────────────

    def interpreter(self, texte: str) -> Intention:
        mots = normaliser(texte).split()
        if not mots:
            return Intention(TypeIntention.AUCUNE, texte=texte, raison="rien entendu")

        # 1. Arrêt : prioritaire, sans mot de réveil.
        if any(mot in MOTS_STOP for mot in mots):
            return Intention(TypeIntention.STOP, texte=texte)

        # 2. Mot de réveil : sans lui, tout le reste est de la conversation.
        if self.mot_reveil:
            positions = [i for i, mot in enumerate(mots) if _proche(mot, self.mot_reveil, SEUIL_VERBE)]
            if not positions:
                return Intention(TypeIntention.AUCUNE, texte=texte, raison="mot de réveil absent")
            mots = mots[positions[0] + 1 :]

        if any(mot in MOTS_QUITTER for mot in mots) or any(
            _contient_phrase(mots, p) for p in PHRASES_QUITTER
        ):
            return Intention(TypeIntention.QUITTER, texte=texte)

        # 3. Saisie : un verbe, pas de négation, au moins un objet connu.
        verbe = next(
            (i for i, mot in enumerate(mots) if any(_proche(mot, v, SEUIL_VERBE) for v in VERBES_SAISIE)),
            None,
        )
        if verbe is not None:
            if any(mot in MOTS_NEGATION for mot in mots[verbe + 1 : verbe + 3]):
                return Intention(TypeIntention.AUCUNE, texte=texte, raison="ordre négatif")
            # L'objet est cherché dans toute la phrase : « la balle, attrape-la » est accepté.
            objets = self._objets_mentionnes(mots)
            if objets:
                return Intention(TypeIntention.ATTRAPER, objets=tuple(objets), texte=texte)
            return Intention(TypeIntention.OBJET_INCONNU, texte=texte, raison="objet non reconnu")

        if any(mot in MOTS_AIDE for mot in mots) or any(_contient_phrase(mots, p) for p in PHRASES_AIDE):
            return Intention(TypeIntention.AIDE, texte=texte)
        return Intention(TypeIntention.AUCUNE, texte=texte, raison="aucun ordre reconnu")
