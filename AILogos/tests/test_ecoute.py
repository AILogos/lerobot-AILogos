"""Détection de parole et écoute micro, avec des signaux synthétiques (aucun micro requis)."""

from __future__ import annotations

import threading

import numpy as np

from AILogos.voix.ecoute import FREQUENCE_HZ, TAILLE_BLOC, DetecteurParole, EcouteMicro

GENERATEUR = np.random.default_rng(0)


def _silence(duree_s: float, niveau: float = 0.001) -> np.ndarray:
    return (GENERATEUR.standard_normal(int(duree_s * FREQUENCE_HZ)) * niveau).astype(np.float32)


def _voix(duree_s: float, amplitude: float = 0.2) -> np.ndarray:
    temps = np.arange(int(duree_s * FREQUENCE_HZ)) / FREQUENCE_HZ
    return (amplitude * np.sin(2 * np.pi * 220 * temps)).astype(np.float32)


def _phrases(detecteur: DetecteurParole, signal: np.ndarray) -> list[np.ndarray]:
    phrases = []
    for debut in range(0, len(signal) - TAILLE_BLOC + 1, TAILLE_BLOC):
        phrase = detecteur.ajouter(signal[debut : debut + TAILLE_BLOC])
        if phrase is not None:
            phrases.append(phrase)
    return phrases


def test_une_phrase_entre_deux_silences():
    signal = np.concatenate([_silence(1.0), _voix(1.0), _silence(1.5)])
    phrases = _phrases(DetecteurParole(silence_fin_s=0.8), signal)
    assert len(phrases) == 1
    # La phrase contient la parole, un peu d'avance, et le silence de fin.
    assert 1.0 <= len(phrases[0]) / FREQUENCE_HZ <= 2.3


def test_deux_phrases_separees():
    signal = np.concatenate([_silence(0.5), _voix(0.8), _silence(1.2), _voix(0.6), _silence(1.2)])
    assert len(_phrases(DetecteurParole(silence_fin_s=0.8), signal)) == 2


def test_une_pause_courte_ne_coupe_pas_la_phrase():
    signal = np.concatenate([_silence(0.5), _voix(0.5), _silence(0.3), _voix(0.5), _silence(1.2)])
    assert len(_phrases(DetecteurParole(silence_fin_s=0.8), signal)) == 1


def test_silence_et_claquement_ne_declenchent_rien():
    assert _phrases(DetecteurParole(), _silence(3.0)) == []
    claquement = np.concatenate([_silence(1.0), _voix(0.05, amplitude=0.8), _silence(1.5)])
    assert _phrases(DetecteurParole(), claquement) == []


def test_phrase_coupee_a_la_duree_maximale():
    signal = np.concatenate([_silence(0.5), _voix(5.0)])
    phrases = _phrases(DetecteurParole(duree_max_s=2.0), signal)
    assert phrases, "une phrase sans fin doit être rendue à la durée maximale"
    assert all(len(p) / FREQUENCE_HZ <= 2.4 for p in phrases)


def test_bruit_continu_releve_le_seuil():
    """Un bruit de moteur continu finit par être traité comme du bruit de fond."""
    detecteur = DetecteurParole(duree_max_s=1.0)
    seuil_initial = detecteur.seuil
    _phrases(detecteur, np.concatenate([_silence(0.5), _silence(3.0, niveau=0.05)]))
    assert detecteur.seuil > 5 * seuil_initial
    # Une voix nettement plus forte que ce bruit reste détectée.
    assert len(_phrases(detecteur, np.concatenate([_voix(1.0, amplitude=0.5), _silence(1.5)]))) == 1


def test_ecoute_micro_transcrit_et_se_met_en_pause():
    """La boucle d'écoute : blocs → phrase → transcription → rappel ; rien pendant ``silence()``."""

    class TranscripteurFactice:
        def transcrire(self, audio: np.ndarray) -> str:
            return f"phrase de {len(audio) / FREQUENCE_HZ:.1f} s"

    recus: list[str] = []
    recu = threading.Event()

    def sur_texte(texte: str) -> None:
        recus.append(texte)
        recu.set()

    ecoute = EcouteMicro(TranscripteurFactice(), sur_texte, DetecteurParole(silence_fin_s=0.3))
    fil = threading.Thread(target=ecoute._boucle, daemon=True)
    fil.start()
    try:
        signal = np.concatenate([_silence(0.3), _voix(0.6), _silence(0.6)])
        for debut in range(0, len(signal) - TAILLE_BLOC + 1, TAILLE_BLOC):
            ecoute._blocs.put(signal[debut : debut + TAILLE_BLOC])
        assert recu.wait(5.0)
        assert recus and recus[0].startswith("phrase de")

        # Pendant que le robot parle, ce que le micro capte est jeté.
        recu.clear()
        with ecoute.silence():
            assert ecoute._sourd == 1
        assert ecoute._sourd == 0 and ecoute._a_vider.is_set()
    finally:
        ecoute._fin.set()
        fil.join(timeout=2.0)
