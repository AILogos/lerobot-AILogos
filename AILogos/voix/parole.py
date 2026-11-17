"""Retour vocal : le robot annonce ce qu'il a compris et ce qu'il fait.

Utilise la synthèse vocale du système quand elle existe (``spd-say`` ou ``espeak-ng``
sous Linux, ``say`` sous macOS, PowerShell sous Windows). Sans elle, le message est
simplement affiché : le retour vocal n'est jamais bloquant pour le programme.
"""

from __future__ import annotations

import platform
import shutil
import subprocess

DELAI_MAX_S = 10


def _commande(texte: str, langue: str) -> list[str] | None:
    systeme = platform.system()
    if systeme == "Darwin" and shutil.which("say"):
        return ["say", texte]
    if systeme == "Linux":
        if shutil.which("spd-say"):
            return ["spd-say", "--wait", "-l", langue, texte]
        for programme in ("espeak-ng", "espeak"):
            if shutil.which(programme):
                return [programme, "-v", langue, texte]
    if systeme == "Windows" and shutil.which("PowerShell"):
        texte = texte.replace("'", "''")
        return [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{texte}')",
        ]
    return None


class Parole:
    """Affiche chaque message, et le prononce si ``actif`` et qu'une synthèse vocale est installée."""

    def __init__(self, actif: bool = True, langue: str = "fr"):
        self.langue = langue
        self.actif = actif and _commande("test", langue) is not None

    def dire(self, texte: str) -> None:
        print(f"🤖 {texte}", flush=True)
        if not self.actif:
            return
        commande = _commande(texte, self.langue)
        if commande is None:
            return
        try:
            subprocess.run(commande, check=False, timeout=DELAI_MAX_S, capture_output=True)  # noqa: S603  # nosec B603
        except (OSError, subprocess.TimeoutExpired):
            self.actif = False  # synthèse défaillante : on continue en texte seul
