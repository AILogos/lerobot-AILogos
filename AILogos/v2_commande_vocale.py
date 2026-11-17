"""V2 — Commander le robot à la voix : « attrape la balle ».

Chaîne complète : micro → détection de parole → transcription Whisper (locale) →
interprétation de l'ordre → politique entraînée pour l'objet demandé → robot.

Exemples (depuis la racine du dépôt) ::

    uv run python -m AILogos.v2_commande_vocale                      # micro + robot
    uv run python -m AILogos.v2_commande_vocale --mode clavier       # ordres tapés au clavier
    uv run python -m AILogos.v2_commande_vocale --simulation         # sans robot (mise au point)
    uv run python -m AILogos.v2_commande_vocale --texte "attrape la balle puis le crayon"
    uv run python -m AILogos.v2_commande_vocale --lister-micros

Ordres compris
    « attrape la balle », « prends le crayon », « ramasse la balle puis le crayon »…
    « stop » / « arrête »   interrompt la saisie en cours (le bras revient au repos)
    « au revoir »           quitte le programme

Sécurité : la voix n'est pas un arrêt d'urgence (il faut 1 à 2 s pour comprendre « stop »).
Ctrl+C interrompt immédiatement la saisie ; gardez l'alimentation du bras à portée de main.
"""

from __future__ import annotations

import contextlib
import queue
import threading
from typing import Any

from . import journal
from .cli import analyseur, charger, configurer_journalisation, executer
from .configuration import Configuration
from .moteur import ErreurExecution, MoteurSaisie
from .voix.ecoute import (
    DetecteurParole,
    EcouteClavier,
    EcouteMicro,
    ErreurVoix,
    TranscripteurWhisper,
    lister_micros,
)
from .voix.interpreteur import Intention, Interpreteur, TypeIntention
from .voix.parole import Parole


class AssistantVocal:
    """Relie les phrases entendues aux saisies du robot.

    ``recevoir`` est appelé par le fil d'écoute pour chaque phrase ; il ne fait que
    décider (arrêter, mettre une saisie en file, répondre). Les saisies elles-mêmes
    s'exécutent dans le fil principal (``boucle``), une à la fois.
    """

    def __init__(
        self,
        cfg: Configuration,
        moteur: MoteurSaisie,
        interpreteur: Interpreteur,
        parole: Parole,
        duree_s: float | None = None,
    ):
        self.cfg = cfg
        self.moteur = moteur
        self.interpreteur = interpreteur
        self.parole = parole
        self.duree_s = duree_s or cfg.execution.duree_s
        self.ecoute: Any = None  # renseignée après construction : sert à couper le micro quand le robot parle
        self._file: queue.Queue[Intention] = queue.Queue()
        self._occupe = threading.Event()
        self._annule = threading.Event()
        self._fin = threading.Event()
        self._arret_courant = threading.Event()

    # ── Côté écoute ──────────────────────────────────────────────────────────

    def recevoir(self, texte: str) -> Intention:
        """Interprète une phrase entendue et y réagit. Retourne l'intention comprise."""
        intention = self.interpreteur.interpreter(texte)
        print(f"🎤 « {texte} »", flush=True)

        if intention.type is TypeIntention.STOP:
            if self._occupe.is_set():
                self._annule.set()
                self._arret_courant.set()
                print("   → arrêt demandé", flush=True)
            else:
                print("   (rien à arrêter)", flush=True)
        elif intention.type is TypeIntention.QUITTER:
            self.demander_fin()
        elif intention.type is TypeIntention.ATTRAPER:
            if self._occupe.is_set():
                print("   (occupé : dites « stop » pour interrompre la saisie en cours)", flush=True)
            else:
                self._occupe.set()
                self._file.put(intention)
        elif intention.type is TypeIntention.OBJET_INCONNU:
            self._repondre(
                f"Je ne connais pas cet objet. Je sais attraper : {self.interpreteur.objets_connus()}."
            )
        elif intention.type is TypeIntention.AIDE:
            self._repondre(
                f"Dites par exemple : attrape la balle. Objets connus : {self.interpreteur.objets_connus()}. "
                "Dites stop pour m'arrêter."
            )
        else:
            print(f"   (ignoré : {intention.raison})", flush=True)
        return intention

    def demander_fin(self) -> None:
        self._fin.set()
        self._annule.set()
        self._arret_courant.set()

    def _repondre(self, texte: str) -> None:
        # Pendant une saisie on ne parle pas : le micro doit rester attentif à « stop ».
        if self._occupe.is_set():
            print(f"🤖 {texte}", flush=True)
        else:
            self._dire(texte)

    def _dire(self, texte: str) -> None:
        silence = self.ecoute.silence() if self.ecoute is not None else contextlib.nullcontext()
        with silence:
            self.parole.dire(texte)

    # ── Côté robot ───────────────────────────────────────────────────────────

    def boucle(self) -> None:
        """Exécute les saisies demandées jusqu'à « au revoir » ou Ctrl+C."""
        while not self._fin.is_set():
            self.traiter_file(attente_s=0.2)

    def traiter_file(self, attente_s: float = 0.0) -> None:
        """Exécute les saisies en attente (une seule passe)."""
        while True:
            try:
                intention = self._file.get(timeout=attente_s) if attente_s else self._file.get_nowait()
            except queue.Empty:
                return
            self._attraper(intention)

    def _attraper(self, intention: Intention) -> None:
        self._annule.clear()
        try:
            for objet in intention.objets:
                if self._annule.is_set():
                    break
                if not self._attraper_objet(objet, intention.texte):
                    break
        finally:
            self._occupe.clear()

    def _attraper_objet(self, objet: str, texte: str) -> bool:
        """Lance une saisie ; retourne ``False`` s'il faut abandonner la suite de la phrase."""
        tache = self.cfg.objet(objet).tache
        try:
            self.moteur.verifier(objet)
        except ErreurExecution as erreur:
            print(f"   {erreur}", flush=True)
            self._dire(f"Je n'ai pas encore appris à attraper : {objet}.")
            return False

        self._dire(f"Compris : {tache.lower()}.")
        self._arret_courant = threading.Event()
        if self._annule.is_set():  # « stop » prononcé pendant l'annonce
            return False
        try:
            resultat = self.moteur.executer(objet, self.duree_s, self._arret_courant)
        except ErreurExecution as erreur:
            print(f"   {erreur}", flush=True)
            self._dire("Je ne peux pas lancer la saisie.")
            journal.ecrire(
                self.cfg.journal_essais, "essai", source="voix", objet=objet, texte=texte, erreur=str(erreur)
            )
            return False

        journal.ecrire(
            self.cfg.journal_essais,
            "essai",
            source="voix",
            objet=objet,
            texte=texte,
            succes=None,  # non noté : la campagne de test notée se fait avec tester.py
            duree_s=round(resultat.duree_s, 2),
            interrompu=resultat.interrompu,
            simulation=resultat.simule,
            erreur=resultat.erreur,
            sante=resultat.sante_apres,
        )
        if resultat.erreur:
            print(f"   {resultat.erreur}", flush=True)
            self._dire("Une erreur est survenue pendant la saisie.")
            return False
        if resultat.interrompu:
            self._dire("Arrêté.")
            return False
        self._dire("Terminé.")
        return True


def _etat_des_objets(cfg: Configuration, moteur: MoteurSaisie) -> None:
    print("Objets connus :")
    for nom, objet in cfg.objets.items():
        if moteur.politique_prete(nom):
            etat = "prêt"
        elif moteur.simulation:
            etat = "politique absente → saisie fictive (simulation)"
        else:
            etat = f"politique absente → uv run python -m AILogos.entrainer --objet {nom}"
        print(f"  • {nom:<10} ({', '.join(objet.synonymes)}) : {etat}")


def principal(argv: list[str] | None = None) -> int:
    parser = analyseur(__doc__ or "")
    parser.add_argument("--mode", choices=["micro", "clavier"], default="micro", help="source des ordres")
    parser.add_argument("--texte", help="exécuter un seul ordre donné en texte, puis quitter")
    parser.add_argument("--simulation", action="store_true", help="aucun matériel : robot simulé")
    parser.add_argument("--duree", type=float, help="durée maximale d'une saisie en secondes")
    parser.add_argument("--afficher", action="store_true", help="visualiser caméras et actions dans Rerun")
    parser.add_argument("--sans-retour-vocal", action="store_true", help="réponses affichées seulement")
    parser.add_argument("--lister-micros", action="store_true", help="afficher les micros disponibles")
    parser.add_argument("--verbeux", action="store_true", help="afficher les journaux détaillés de LeRobot")
    args = parser.parse_args(argv)
    cfg = charger(args)
    configurer_journalisation(args.verbeux)

    try:
        if args.lister_micros:
            print("\n".join(lister_micros()) or "Aucun micro détecté.")
            return 0

        au_micro = args.mode == "micro" and not args.texte
        moteur = MoteurSaisie(cfg, simulation=args.simulation, afficher=args.afficher)
        # Le mot de réveil ne sert qu'à l'oral : au clavier, chaque ligne est un ordre voulu.
        interpreteur = Interpreteur.depuis_configuration(cfg)
        if not au_micro:
            interpreteur.mot_reveil = ""
        parole = Parole(cfg.voix.retour_vocal and not args.sans_retour_vocal, cfg.voix.langue)
        assistant = AssistantVocal(cfg, moteur, interpreteur, parole, args.duree)
        _etat_des_objets(cfg, moteur)

        if args.texte:
            assistant.recevoir(args.texte)
            assistant.traiter_file()
            moteur.fermer()
            return 0

        if au_micro:
            print(f"\nChargement du modèle Whisper « {cfg.voix.modele} » ({cfg.voix.device})…", flush=True)
            transcripteur = TranscripteurWhisper(
                cfg.voix.modele,
                cfg.voix.langue,
                cfg.voix.device,
                cfg.voix.precision,
                amorce=interpreteur.phrase_amorce(),
            )
            detecteur = DetecteurParole(cfg.voix.seuil_parole, cfg.voix.silence_fin_s, cfg.voix.duree_max_s)
            ecoute: Any = EcouteMicro(transcripteur, assistant.recevoir, detecteur, cfg.voix.micro)
        else:
            ecoute = EcouteClavier(assistant.recevoir, sur_fin=assistant.demander_fin)
        assistant.ecoute = ecoute
        # Charge torch et LeRobot maintenant plutôt qu'au premier ordre : la première saisie démarre plus vite.
        moteur.prechauffer()

        reveil = f"« {cfg.voix.mot_reveil} », puis " if au_micro and cfg.voix.mot_reveil else ""
        action = "Dites" if au_micro else "Tapez"
        print(
            f"\n{action} {reveil}« attrape la balle », « stop », « au revoir ».  Ctrl+C pour quitter."
            + ("\nMode simulation : aucun moteur ne bouge." if args.simulation else "")
            + "\nLe bras doit être en position de repos avant chaque ordre.\n",
            flush=True,
        )
        ecoute.demarrer()
    except ErreurVoix as erreur:
        print(f"\n[voix] {erreur}")
        return 1

    try:
        assistant.boucle()
    except KeyboardInterrupt:
        print("\nArrêt demandé.")
    finally:
        ecoute.arreter()
        moteur.fermer()
    return 0


if __name__ == "__main__":
    executer(principal)
