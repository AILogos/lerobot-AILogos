"""Interprétation des ordres vocaux : du texte transcrit à l'intention."""

from __future__ import annotations

import pytest

from AILogos.voix.interpreteur import Interpreteur, TypeIntention, normaliser

OBJETS = {"balle": ["balle", "ballon", "boule"], "crayon": ["crayon", "stylo", "feutre"]}


@pytest.fixture
def interpreteur() -> Interpreteur:
    return Interpreteur(OBJETS)


def test_normaliser():
    assert (
        normaliser("  N'attrape PAS le crayon, s'il te plaît !  ") == "n attrape pas le crayon s il te plait"
    )


@pytest.mark.parametrize(
    ("phrase", "objets"),
    [
        ("attrape la balle", ("balle",)),
        ("Attrape la balle.", ("balle",)),
        ("Peux-tu attraper le crayon, s'il te plaît ?", ("crayon",)),
        ("prends le stylo", ("crayon",)),
        ("ramasse le ballon", ("balle",)),
        ("Récupère la boule rouge", ("balle",)),
        ("donne-moi le feutre", ("crayon",)),
        ("attrape les balles", ("balle",)),
        ("la balle, attrape-la", ("balle",)),
        # erreurs de transcription courantes
        ("atrape la bale", ("balle",)),
        ("attrappe le craion", ("crayon",)),
        # plusieurs objets, dans l'ordre demandé
        ("attrape la balle puis le crayon", ("balle", "crayon")),
        ("prends le crayon et ensuite la balle", ("crayon", "balle")),
        # anglais toléré
        ("grab the balle", ("balle",)),
    ],
)
def test_ordres_de_saisie(interpreteur, phrase, objets):
    intention = interpreteur.interpreter(phrase)
    assert intention.type is TypeIntention.ATTRAPER
    assert intention.objets == objets
    assert intention.texte == phrase


@pytest.mark.parametrize(
    "phrase",
    [
        "stop",
        "Stop !",
        "arrête",
        "Arrête-toi tout de suite",
        "arrêtez",
        "halte",
        "stop, n'attrape pas la balle",
    ],
)
def test_arret_prioritaire(interpreteur, phrase):
    assert interpreteur.interpreter(phrase).type is TypeIntention.STOP


@pytest.mark.parametrize(
    "phrase",
    [
        "",
        "   ",
        "bonjour tout le monde",
        "la balle est sur la table",  # objet sans verbe de saisie : pas un ordre
        "il fait beau aujourd'hui",
        "n'attrape pas la balle",
        "ne prends jamais le crayon",
        "c'est quelque chose, cette balle",  # « chose » ne doit pas passer pour « chope »
        "Sous-titres réalisés par la communauté d'Amara.org",  # phrase typique inventée par Whisper
    ],
)
def test_phrases_sans_ordre(interpreteur, phrase):
    assert interpreteur.interpreter(phrase).type is TypeIntention.AUCUNE


@pytest.mark.parametrize("phrase", ["attrape la tasse", "prends ça", "ramasse le téléphone"])
def test_objet_inconnu(interpreteur, phrase):
    assert interpreteur.interpreter(phrase).type is TypeIntention.OBJET_INCONNU


@pytest.mark.parametrize("phrase", ["au revoir", "quitte", "Au revoir le robot !", "fin du programme"])
def test_quitter(interpreteur, phrase):
    assert interpreteur.interpreter(phrase).type is TypeIntention.QUITTER


@pytest.mark.parametrize("phrase", ["aide", "que sais-tu faire ?", "quels objets connais-tu ?"])
def test_aide(interpreteur, phrase):
    assert interpreteur.interpreter(phrase).type is TypeIntention.AIDE


def test_mot_de_reveil():
    interpreteur = Interpreteur(OBJETS, mot_reveil="Robot")
    assert interpreteur.interpreter("attrape la balle").type is TypeIntention.AUCUNE
    assert interpreteur.interpreter("au revoir").type is TypeIntention.AUCUNE

    intention = interpreteur.interpreter("Robot, attrape la balle")
    assert intention.type is TypeIntention.ATTRAPER and intention.objets == ("balle",)
    # Ce qui précède le mot de réveil est ignoré.
    intention = interpreteur.interpreter("prends le crayon... non, robot, attrape la balle")
    assert intention.objets == ("balle",)
    # L'arrêt n'a jamais besoin du mot de réveil.
    assert interpreteur.interpreter("stop").type is TypeIntention.STOP


def test_synonyme_en_plusieurs_mots():
    interpreteur = Interpreteur({"balle": ["balle de tennis"], "tasse": ["tasse à café", "mug"]})
    assert interpreteur.interpreter("attrape la balle de tennis").objets == ("balle",)
    assert interpreteur.interpreter("prends la tasse à café").objets == ("tasse",)
    assert interpreteur.interpreter("prends le mug").objets == ("tasse",)


def test_phrase_amorce_contient_le_vocabulaire(cfg):
    amorce = Interpreteur.depuis_configuration(cfg).phrase_amorce()
    for attendu in ("Attrape la balle.", "Attrape le crayon.", "ballon", "stylo", "Stop"):
        assert attendu in amorce


def test_depuis_configuration(cfg):
    interpreteur = Interpreteur.depuis_configuration(cfg)
    assert interpreteur.interpreter("attrape le ballon").objets == ("balle",)
    assert interpreteur.objets_connus() == "balle, crayon"
