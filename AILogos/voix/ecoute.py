"""Écoute : du micro (ou du clavier) vers du texte.

Trois briques indépendantes :

* ``DetecteurParole`` découpe le flux audio en phrases, d'après le niveau sonore.
  Il ne dépend que de numpy et se teste avec des signaux synthétiques.
* ``TranscripteurWhisper`` transcrit une phrase avec faster-whisper, en local :
  aucun son ne quitte l'ordinateur.
* ``EcouteMicro`` relie le micro, le détecteur et le transcripteur dans un fil
  d'arrière-plan, et appelle ``sur_texte(phrase)`` à chaque phrase comprise.
  ``EcouteClavier`` fait la même chose à partir de lignes tapées au clavier.

Dépendances de la partie micro : ``uv pip install -r AILogos/requirements-voix.txt``.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any, Protocol

import numpy as np

FREQUENCE_HZ = 16_000  # fréquence attendue par Whisper
DUREE_BLOC_S = 0.03
TAILLE_BLOC = int(FREQUENCE_HZ * DUREE_BLOC_S)


class ErreurVoix(RuntimeError):
    """La chaîne vocale ne peut pas démarrer (dépendance, micro ou modèle manquant)."""


# ═════════════════════════════════════════════════════════════════════════════
# Détection de parole
# ═════════════════════════════════════════════════════════════════════════════


class DetecteurParole:
    """Découpe un flux audio mono en phrases, par seuil d'énergie adaptatif.

    Le bruit ambiant est mesuré en continu pendant les silences ; la parole est ce qui
    dépasse ``seuil_parole`` fois ce bruit. Une phrase se termine après ``silence_fin_s``
    de silence, ou au bout de ``duree_max_s``.
    """

    def __init__(
        self,
        seuil_parole: float = 2.5,
        silence_fin_s: float = 0.8,
        duree_max_s: float = 6.0,
        duree_min_s: float = 0.3,
        niveau_min: float = 0.004,
        frequence_hz: int = FREQUENCE_HZ,
    ):
        self.seuil_parole = seuil_parole
        self.silence_fin_s = silence_fin_s
        self.duree_max_s = duree_max_s
        self.duree_min_s = duree_min_s
        self.niveau_min = niveau_min  # plancher : un micro parfaitement silencieux ne déclenche pas
        self.frequence_hz = frequence_hz
        self.bruit = niveau_min
        self.reinitialiser()

    def reinitialiser(self) -> None:
        """Oublie la phrase en cours (après une pause de l'écoute, par exemple)."""
        self._avant: deque[np.ndarray] = deque(maxlen=10)  # ~0,3 s gardées avant le début détecté
        self._phrase: list[np.ndarray] = []
        self._niveaux: list[float] = []
        self._en_parole = False
        self._blocs_forts = 0
        self._silence_s = 0.0
        self._parole_s = 0.0

    @property
    def seuil(self) -> float:
        return max(self.niveau_min, self.bruit * self.seuil_parole)

    @property
    def en_parole(self) -> bool:
        return self._en_parole

    def ajouter(self, bloc: np.ndarray) -> np.ndarray | None:
        """Ajoute un bloc audio (float32 entre -1 et 1) ; retourne une phrase quand elle se termine."""
        bloc = np.asarray(bloc, dtype=np.float32).reshape(-1)
        if bloc.size == 0:
            return None
        duree = bloc.size / self.frequence_hz
        niveau = float(np.sqrt(np.mean(bloc**2)))
        fort = niveau > self.seuil

        if not self._en_parole:
            self._avant.append(bloc)
            if fort:
                self._blocs_forts += 1
            else:
                self._blocs_forts = 0
                # Le bruit de fond n'est mis à jour que hors parole, lentement.
                self.bruit = 0.95 * self.bruit + 0.05 * max(niveau, self.niveau_min / self.seuil_parole)
            # Trois blocs forts de suite (~90 ms) : un claquement isolé ne démarre pas une phrase.
            if self._blocs_forts >= 3:
                self._en_parole = True
                self._phrase = list(self._avant)
                self._parole_s = sum(b.size for b in self._phrase) / self.frequence_hz
                self._silence_s = 0.0
            return None

        self._phrase.append(bloc)
        self._niveaux.append(niveau)
        self._parole_s += duree
        self._silence_s = 0.0 if fort else self._silence_s + duree
        if self._silence_s < self.silence_fin_s and self._parole_s < self.duree_max_s:
            return None

        phrase = np.concatenate(self._phrase)
        duree_utile = self._parole_s - self._silence_s
        if self._silence_s < self.silence_fin_s:
            # Aucun silence pendant toute la durée maximale : un bruit continu (moteurs,
            # ventilateur) est pris pour de la parole. On relève le bruit de fond à son niveau.
            self.bruit = max(self.bruit, float(np.percentile(self._niveaux, 20)))
        self.reinitialiser()
        return phrase if duree_utile >= self.duree_min_s else None


# ═════════════════════════════════════════════════════════════════════════════
# Transcription
# ═════════════════════════════════════════════════════════════════════════════


class Transcripteur(Protocol):
    def transcrire(self, audio: np.ndarray) -> str: ...


class TranscripteurWhisper:
    """Transcription locale avec faster-whisper.

    Le modèle est téléchargé depuis le Hub Hugging Face au premier lancement, puis
    gardé en cache : les lancements suivants fonctionnent hors ligne.
    """

    def __init__(
        self,
        modele: str = "small",
        langue: str = "fr",
        device: str = "cpu",
        precision: str = "int8",
        amorce: str | None = None,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as erreur:
            raise ErreurVoix(
                "faster-whisper n'est pas installé. Installez les dépendances vocales :\n"
                "  uv pip install -r AILogos/requirements-voix.txt"
            ) from erreur
        self.langue = langue
        self.amorce = amorce
        try:
            self._modele = WhisperModel(modele, device=device, compute_type=precision)
        except Exception as erreur:
            raise ErreurVoix(
                f"Impossible de charger le modèle Whisper « {modele} » sur {device} ({precision}) : {erreur}\n"
                "Premier lancement : une connexion internet est nécessaire pour le télécharger."
            ) from erreur

    def transcrire(self, audio: np.ndarray) -> str:
        segments, _ = self._modele.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.langue,
            initial_prompt=self.amorce,
            beam_size=5,
            # Écarte le bruit des moteurs et limite les phrases inventées sur du silence.
            vad_filter=True,
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()


# ═════════════════════════════════════════════════════════════════════════════
# Sources de commandes
# ═════════════════════════════════════════════════════════════════════════════


def _importer_sounddevice() -> Any:
    try:
        import sounddevice
    except ImportError as erreur:
        raise ErreurVoix(
            "sounddevice n'est pas installé. Installez les dépendances vocales :\n"
            "  uv pip install -r AILogos/requirements-voix.txt"
        ) from erreur
    except OSError as erreur:
        raise ErreurVoix(
            f"La bibliothèque audio PortAudio est introuvable ({erreur}).\n"
            "Linux : sudo apt install libportaudio2"
        ) from erreur
    return sounddevice


def lister_micros() -> list[str]:
    """Périphériques d'entrée audio, sous la forme « index : nom »."""
    sounddevice = _importer_sounddevice()
    return [
        f"{index} : {peripherique['name']}"
        for index, peripherique in enumerate(sounddevice.query_devices())
        if peripherique["max_input_channels"] > 0
    ]


class EcouteMicro:
    """Écoute le micro en continu et appelle ``sur_texte`` pour chaque phrase transcrite."""

    def __init__(
        self,
        transcripteur: Transcripteur,
        sur_texte: Callable[[str], None],
        detecteur: DetecteurParole | None = None,
        micro: int | str | None = None,
    ):
        self.transcripteur = transcripteur
        self.sur_texte = sur_texte
        self.detecteur = detecteur or DetecteurParole()
        self.micro = micro
        self._blocs: queue.Queue[np.ndarray] = queue.Queue()
        self._fin = threading.Event()
        self._a_vider = threading.Event()
        self._sourd = 0  # compteur de pauses imbriquées
        self._verrou = threading.Lock()
        self._flux: Any = None
        self._fil: threading.Thread | None = None

    # ── Pause pendant que le robot parle (sinon il s'entendrait lui-même) ────

    @contextlib.contextmanager
    def silence(self) -> Iterator[None]:
        """Suspend l'écoute le temps du bloc ``with``."""
        with self._verrou:
            self._sourd += 1
        try:
            yield
        finally:
            with self._verrou:
                self._sourd -= 1
            self._vider()

    def _vider(self) -> None:
        # Le détecteur appartient au fil d'écoute : on lui demande de repartir de zéro.
        self._a_vider.set()

    # ── Cycle de vie ─────────────────────────────────────────────────────────

    def demarrer(self) -> None:
        sounddevice = _importer_sounddevice()

        def rappel(donnees: np.ndarray, _nombre: int, _temps: Any, _etat: Any) -> None:
            if not self._sourd:
                self._blocs.put(donnees[:, 0].copy())

        try:
            self._flux = sounddevice.InputStream(
                samplerate=FREQUENCE_HZ,
                blocksize=TAILLE_BLOC,
                channels=1,
                dtype="float32",
                device=self.micro,
                callback=rappel,
            )
            self._flux.start()
        except Exception as erreur:
            raise ErreurVoix(
                f"Impossible d'ouvrir le micro ({erreur}).\n"
                "Listez les micros avec --lister-micros puis renseignez « voix.micro » dans config.yaml."
            ) from erreur
        self._fil = threading.Thread(target=self._boucle, name="ailogos-ecoute", daemon=True)
        self._fil.start()

    def _boucle(self) -> None:
        while not self._fin.is_set():
            if self._a_vider.is_set():
                self._a_vider.clear()
                with contextlib.suppress(queue.Empty):
                    while True:
                        self._blocs.get_nowait()
                self.detecteur.reinitialiser()
            try:
                bloc = self._blocs.get(timeout=0.2)
            except queue.Empty:
                continue
            phrase = self.detecteur.ajouter(bloc)
            if phrase is None:
                continue
            try:
                texte = self.transcripteur.transcrire(phrase)
            except Exception as erreur:  # noqa: BLE001 — une phrase ratée ne doit pas couper l'écoute
                print(f"[voix] transcription impossible : {erreur}", file=sys.stderr)
                continue
            if texte:
                self.sur_texte(texte)

    def arreter(self) -> None:
        self._fin.set()
        if self._flux is not None:
            self._flux.stop()
            self._flux.close()
            self._flux = None
        if self._fil is not None and self._fil is not threading.current_thread():
            self._fil.join(timeout=2.0)


class EcouteClavier:
    """Même interface que ``EcouteMicro``, mais les ordres sont tapés au clavier.

    Pratique pour mettre au point sans micro, ou dans un atelier bruyant.
    """

    def __init__(self, sur_texte: Callable[[str], None], sur_fin: Callable[[], None] | None = None):
        self.sur_texte = sur_texte
        self.sur_fin = sur_fin
        self._fil: threading.Thread | None = None

    @contextlib.contextmanager
    def silence(self) -> Iterator[None]:
        yield

    def demarrer(self) -> None:
        self._fil = threading.Thread(target=self._boucle, name="ailogos-clavier", daemon=True)
        self._fil.start()

    def _boucle(self) -> None:
        for ligne in sys.stdin:
            if ligne.strip():
                self.sur_texte(ligne.strip())
        # Fin de l'entrée (Ctrl+D, ou fin d'un fichier redirigé) : plus aucun ordre ne viendra.
        if self.sur_fin is not None:
            self.sur_fin()

    def arreter(self) -> None:
        pass
