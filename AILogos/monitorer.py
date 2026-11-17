"""Étape 4 — Monitorer le robot, l'entraînement et les résultats.

Quatre vues (depuis la racine du dépôt) ::

    uv run python -m AILogos.monitorer                       # vue d'ensemble : où en est chaque objet
    uv run python -m AILogos.monitorer robot                 # moteurs et caméras en direct
    uv run python -m AILogos.monitorer robot --csv releve.csv
    uv run python -m AILogos.monitorer entrainement --objet balle --suivre
    uv run python -m AILogos.monitorer bilan                 # taux de réussite et historique

``robot`` ouvre le port série en lecture seule (aucun couple appliqué, aucun réglage
modifié) : le lancer quand aucun autre programme n'utilise le bras. Pendant une saisie,
la visualisation se fait avec ``--afficher`` (Rerun) dans tester.py ou la commande vocale.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import journal
from .cli import analyseur, charger, executer
from .commandes import images_du_dataset
from .configuration import Configuration
from .sante import ALERTE, CRITIQUE, EtatRobot, lire_sante

_COULEURS = {"ok": "\033[32m", ALERTE: "\033[33m", CRITIQUE: "\033[31m"}
_FIN = "\033[0m"
_BLOCS = "▁▂▃▄▅▆▇█"


def _couleur(texte: str, niveau: str) -> str:
    return f"{_COULEURS.get(niveau, '')}{texte}{_FIN}" if sys.stdout.isatty() else texte


def _valeur(valeur: float | None, format_: str, unite: str = "") -> str:
    return "—" if valeur is None else f"{valeur:{format_}}{unite}"


# ═════════════════════════════════════════════════════════════════════════════
# Vue « robot » : télémétrie en direct
# ═════════════════════════════════════════════════════════════════════════════


def tableau_sante(etat: EtatRobot) -> str:
    lignes = [f"{'moteur':<15}{'position':>10}{'temp.':>9}{'tension':>10}{'charge':>9}{'courant':>11}  état"]
    for m in etat.moteurs:
        lignes.append(
            f"{m.nom:<15}{_valeur(m.position, '8.1f'):>10}{_valeur(m.temperature_c, '.0f', ' °C'):>9}"
            f"{_valeur(m.tension_v, '.1f', ' V'):>10}{_valeur(m.charge_pct, '.0f', ' %'):>9}"
            f"{_valeur(m.courant_ma, '.0f', ' mA'):>11}  "
            + _couleur("; ".join(m.messages) if m.messages else "ok", m.niveau)
        )
    return "\n".join(lignes)


def _etat_cameras(cameras: dict[str, Any]) -> list[str]:
    lignes = []
    for nom, camera in cameras.items():
        debut = time.perf_counter()
        try:
            image = camera.async_read(timeout_ms=1000)
            attente_ms = (time.perf_counter() - debut) * 1e3
            luminosite = float(image.mean())
            remarque = "  ⚠ image très sombre (cache, éclairage ?)" if luminosite < 15 else ""
            lignes.append(
                f"{nom:<15}{image.shape[1]}×{image.shape[0]}  image reçue en {attente_ms:4.0f} ms  "
                f"luminosité {luminosite:5.1f}/255{remarque}"
            )
        except Exception as erreur:  # noqa: BLE001 — une caméra en panne ne doit pas arrêter le suivi
            lignes.append(_couleur(f"{nom:<15}pas d'image : {erreur}", CRITIQUE))
    return lignes


def vue_robot(cfg: Configuration, args: Any) -> int:
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(cfg.config_robot())
    bus = getattr(robot, "bus", None)
    if bus is None:
        print(f"Le robot « {cfg.robot['type']} » n'expose pas de bus de moteurs à surveiller.")
        return 1
    cameras = {} if args.sans_cameras else dict(getattr(robot, "cameras", {}))

    print(f"Connexion au bras sur {cfg.robot.get('port')} (lecture seule)…")
    bus.connect()
    fichier_csv = ecrivain = None
    try:
        for nom, camera in cameras.items():
            try:
                camera.connect()
            except Exception as erreur:  # noqa: BLE001
                print(f"Caméra « {nom} » indisponible : {erreur}")
        cameras = {nom: camera for nom, camera in cameras.items() if camera.is_connected}

        if args.csv:
            nouveau = not Path(args.csv).exists()
            fichier_csv = open(args.csv, "a", newline="", encoding="utf-8")  # noqa: SIM115
            ecrivain = csv.writer(fichier_csv)
            if nouveau:
                ecrivain.writerow(
                    [
                        "date",
                        "moteur",
                        "position",
                        "temperature_c",
                        "tension_v",
                        "charge_pct",
                        "courant_ma",
                        "niveau",
                    ]
                )

        while True:
            etat = lire_sante(robot, cfg.sante)
            if etat is None:
                print("Lecture des moteurs impossible.")
                return 1
            if sys.stdout.isatty() and not args.une_fois:
                print("\033[H\033[J", end="")
            print(f"AILogos · moteurs du bras · {time.strftime('%H:%M:%S')} · Ctrl+C pour quitter\n")
            print(tableau_sante(etat))
            if cameras:
                print("\ncaméras")
                print("\n".join(_etat_cameras(cameras)))
            print("\nÉtat général : " + _couleur(etat.niveau.upper(), etat.niveau))
            if ecrivain is not None and fichier_csv is not None:
                date = time.strftime("%Y-%m-%dT%H:%M:%S")
                for m in etat.moteurs:
                    ecrivain.writerow(
                        [
                            date,
                            m.nom,
                            m.position,
                            m.temperature_c,
                            m.tension_v,
                            m.charge_pct,
                            m.courant_ma,
                            m.niveau,
                        ]
                    )
                fichier_csv.flush()
            if args.une_fois:
                return 0 if etat.niveau != CRITIQUE else 1
            time.sleep(args.intervalle)
    except KeyboardInterrupt:
        return 0
    finally:
        if fichier_csv is not None:
            fichier_csv.close()
        for camera in cameras.values():
            if camera.is_connected:
                camera.disconnect()
        # Lecture seule : on laisse le couple dans l'état où on l'a trouvé.
        bus.disconnect(disable_torque=False)


# ═════════════════════════════════════════════════════════════════════════════
# Vue « entrainement » : progression lue dans le journal écrit par entrainer.py
# ═════════════════════════════════════════════════════════════════════════════

_RE_TOTAL = re.compile(r"cfg\.steps=(\d+)")
_RE_BARRE = re.compile(r"(\d+)/(\d+) \[")
_RE_METRIQUE = re.compile(r"\b(step|loss|l1_loss|lr|step_s|epch):(\S+)")
_RE_VALIDATION = re.compile(r"step (\d+): eval_loss=([\d.eE+-]+)")
_SUFFIXES = {"K": 1e3, "M": 1e6, "B": 1e9}


def _nombre(texte: str) -> float:
    if texte and texte[-1] in _SUFFIXES:
        return float(texte[:-1]) * _SUFFIXES[texte[-1]]
    return float(texte)


@dataclass
class EtatEntrainement:
    total: int | None = None
    pas: int = 0
    epoque: float | None = None
    pertes: list[float] = field(default_factory=list)
    perte_l1: float | None = None
    taux_apprentissage: float | None = None
    secondes_par_pas: float | None = None
    validations: list[tuple[int, float]] = field(default_factory=list)
    checkpoints: int = 0
    termine: bool = False
    erreur: str | None = None

    @property
    def avancement(self) -> float | None:
        return min(self.pas / self.total, 1.0) if self.total else None

    @property
    def restant_s(self) -> float | None:
        if self.total is None or self.secondes_par_pas is None or self.termine:
            return None
        return max(self.total - self.pas, 0) * self.secondes_par_pas


def analyser_journal(texte: str) -> EtatEntrainement:
    """Extrait la progression du dernier entraînement présent dans le journal."""
    # Chaque lancement commence par une ligne « ===== commande ===== » : on garde le dernier.
    texte = texte.rsplit("\n===== ", 1)[-1]
    etat = EtatEntrainement()
    for ligne in texte.splitlines():
        if total := _RE_TOTAL.search(ligne):
            etat.total = int(total.group(1))
        if validation := _RE_VALIDATION.search(ligne):
            etat.validations.append((int(validation.group(1)), float(validation.group(2))))
        if "Checkpoint policy after step" in ligne:
            etat.checkpoints += 1
        if "End of training" in ligne:
            etat.termine = True
        if ligne.startswith("Traceback (most recent call last)"):
            etat.erreur = "une erreur Python a interrompu l'entraînement (voir la fin du journal)"

        metriques = dict(_RE_METRIQUE.findall(ligne))
        if "loss" not in metriques or "step" not in metriques:
            continue
        try:
            # Le pas exact vient de la barre de progression ; sinon de « step:12K », arrondi.
            barre = _RE_BARRE.search(ligne)
            etat.pas = int(barre.group(1)) if barre else int(_nombre(metriques["step"]))
            etat.pertes.append(float(metriques["loss"]))
            etat.perte_l1 = float(metriques["l1_loss"]) if "l1_loss" in metriques else None
            etat.taux_apprentissage = float(metriques["lr"]) if "lr" in metriques else None
            etat.secondes_par_pas = float(metriques["step_s"]) if "step_s" in metriques else None
            etat.epoque = float(metriques["epch"]) if "epch" in metriques else None
        except ValueError:
            continue
    if etat.termine and etat.total:
        etat.pas = etat.total
    return etat


def courbe(valeurs: list[float], largeur: int = 50) -> str:
    """Mini-courbe en caractères (la perte doit descendre de gauche à droite)."""
    if not valeurs:
        return ""
    if len(valeurs) > largeur:
        pas = len(valeurs) / largeur
        valeurs = [valeurs[int(i * pas)] for i in range(largeur)]
    mini, maxi = min(valeurs), max(valeurs)
    etendue = (maxi - mini) or 1.0
    return "".join(_BLOCS[int((v - mini) / etendue * (len(_BLOCS) - 1))] for v in valeurs)


def _duree(secondes: float) -> str:
    minutes, secondes = divmod(int(secondes), 60)
    heures, minutes = divmod(minutes, 60)
    return f"{heures} h {minutes:02d} min" if heures else f"{minutes} min {secondes:02d} s"


def rapport_entrainement(objet: str, etat: EtatEntrainement) -> str:
    lignes = [f"Entraînement « {objet} »"]
    if etat.avancement is not None:
        plein = int(etat.avancement * 30)
        lignes.append(
            f"  [{'█' * plein}{'░' * (30 - plein)}] {etat.avancement:.0%}  pas {etat.pas}/{etat.total}"
        )
    else:
        lignes.append(f"  pas {etat.pas} (total inconnu : entraînement en cours de démarrage ?)")
    if etat.pertes:
        detail = f"  perte : {etat.pertes[-1]:.3f} (début : {etat.pertes[0]:.3f})"
        if etat.perte_l1 is not None:
            detail += f" · erreur L1 sur les actions : {etat.perte_l1:.3f}"
        lignes += [detail, f"  {courbe(etat.pertes)}"]
    if etat.validations:
        pas, valeur = etat.validations[-1]
        meilleur_pas, meilleure = min(etat.validations, key=lambda v: v[1])
        lignes.append(
            f"  validation : {valeur:.4f} au pas {pas} (meilleure : {meilleure:.4f} au pas {meilleur_pas})"
        )
        if len(etat.validations) >= 3 and meilleur_pas != pas and valeur > 1.1 * meilleure:
            lignes.append(
                "  ⚠ la perte de validation remonte : sur-apprentissage possible, inutile de prolonger."
            )
    if etat.epoque is not None:
        lignes.append(f"  époque {etat.epoque:.1f} · checkpoints écrits : {etat.checkpoints}")
    if etat.restant_s is not None and etat.secondes_par_pas is not None:
        lignes.append(
            f"  vitesse : {1 / etat.secondes_par_pas:.1f} pas/s · temps restant ≈ {_duree(etat.restant_s)}"
        )
    if etat.erreur:
        lignes.append(f"  ✗ {etat.erreur}")
    elif etat.termine:
        lignes.append("  ✓ entraînement terminé")
    return "\n".join(lignes)


def vue_entrainement(cfg: Configuration, args: Any) -> int:
    objets = [cfg.objet(args.objet).nom] if args.objet else list(cfg.objets)
    try:
        while True:
            rapports, en_cours = [], False
            for objet in objets:
                chemin = cfg.journal_entrainement(objet)
                if not chemin.exists():
                    rapports.append(f"Entraînement « {objet} »\n  pas encore lancé ({chemin} absent)")
                    continue
                etat = analyser_journal(chemin.read_text(encoding="utf-8", errors="replace"))
                en_cours |= not (etat.termine or etat.erreur)
                rapports.append(rapport_entrainement(objet, etat))
            if args.suivre and sys.stdout.isatty():
                print("\033[H\033[J", end="")
            print("\n\n".join(rapports))
            if not args.suivre or not en_cours:
                return 0
            print(f"\n(actualisation toutes les {args.intervalle:g} s · Ctrl+C pour quitter)")
            time.sleep(args.intervalle)
    except KeyboardInterrupt:
        return 0


# ═════════════════════════════════════════════════════════════════════════════
# Vues « bilan » et « ensemble »
# ═════════════════════════════════════════════════════════════════════════════


def rapport_bilan(evenements: list[dict[str, Any]], derniers: int = 10) -> str:
    essais = [e for e in evenements if e.get("type") == "essai"]
    if not essais:
        return "Aucun essai enregistré pour l'instant (lancez tester.py ou la commande vocale)."
    lignes = [f"{'objet':<14}{'essais':>8}{'réussis':>9}{'taux':>7}{'durée moy.':>12}   dernier essai"]
    for ligne in sorted(journal.bilan(essais).values(), key=lambda b: b.objet):
        lignes.append(
            f"{ligne.objet:<14}{ligne.essais:>8}{ligne.reussites:>9}{_valeur(ligne.taux, '.0%'):>7}"
            f"{_valeur(ligne.duree_moyenne_s, '.1f', ' s'):>12}   {ligne.dernier.replace('T', ' ')}"
        )
    lignes.append(f"\n{min(derniers, len(essais))} derniers essais")
    for e in essais[-derniers:]:
        note = {True: "réussi", False: "raté  ", None: "—     "}[e.get("succes")]
        origine = "voix" if e.get("source") == "voix" else "test"
        details = [e.get("texte") and f"« {e['texte']} »", e.get("commentaire"), e.get("erreur") and "erreur"]
        if e.get("simulation"):
            details.append("simulation")
        suffixe = "  " + " · ".join(str(d) for d in details if d) if any(details) else ""
        lignes.append(
            f"  {str(e.get('date', '')).replace('T', ' ')}  {origine}  {str(e.get('objet')):<10} {note} "
            f"{float(e.get('duree_s') or 0):5.1f} s{suffixe}"
        )
    return "\n".join(lignes)


def vue_bilan(cfg: Configuration, args: Any) -> int:
    evenements = journal.lire(cfg.journal_essais)
    if args.objet:
        nom = cfg.objet(args.objet).nom
        evenements = [e for e in evenements if e.get("objet") == nom]
    print(rapport_bilan(evenements, args.derniers))
    return 0


def _episodes_du_dataset(cfg: Configuration, objet: str) -> int | None:
    try:
        info = json.loads((cfg.racine_dataset(objet) / "meta" / "info.json").read_text(encoding="utf-8"))
        return int(info["total_episodes"])
    except (OSError, ValueError, KeyError):
        return None


def vue_ensemble(cfg: Configuration, args: Any) -> int:
    """Où en est chaque objet : données, politique, résultats — et la prochaine commande à lancer."""
    bilans = journal.bilan(journal.lire(cfg.journal_essais))
    print(f"AILogos · configuration : {cfg.chemin}\n")
    for objet in cfg.objets:
        print(f"● {objet} — « {cfg.objet(objet).tache} »")
        episodes, images = _episodes_du_dataset(cfg, objet), images_du_dataset(cfg, objet)
        if episodes is None:
            print("    données     : aucune démonstration enregistrée")
            suite = f"enregistrer --objet {objet}"
        else:
            print(f"    données     : {episodes} épisodes, {images} images ({cfg.repo_dataset(objet)})")
            suite = f"entrainer --objet {objet}"

        chemin_journal = cfg.journal_entrainement(objet)
        if cfg.politique_disponible(objet):
            print(f"    politique   : prête ({cfg.chemin_politique(objet)})")
            suite = f"tester --objet {objet}"
        elif chemin_journal.exists():
            etat = analyser_journal(chemin_journal.read_text(encoding="utf-8", errors="replace"))
            avancement = f"{etat.avancement:.0%}" if etat.avancement is not None else "démarrage"
            print(f"    politique   : entraînement en cours ou interrompu ({avancement})")
            suite = f"monitorer entrainement --objet {objet} --suivre"
        else:
            print("    politique   : pas encore entraînée")

        resultat = bilans.get(objet)
        if resultat and resultat.taux is not None:
            print(f"    tests       : {resultat.reussites}/{resultat.essais} réussis ({resultat.taux:.0%})")
            if cfg.politique_disponible(objet):
                suite = "v2_commande_vocale"
        else:
            print("    tests       : aucun essai noté")
        print(f"    à faire     : uv run python -m AILogos.{suite}\n")
    return 0


def principal(argv: list[str] | None = None) -> int:
    parser = analyseur(__doc__ or "")
    vues = parser.add_subparsers(dest="vue")

    robot = vues.add_parser("robot", help="moteurs (température, tension, charge) et caméras en direct")
    robot.add_argument("--intervalle", type=float, default=1.0, help="secondes entre deux relevés")
    robot.add_argument("--csv", help="ajouter chaque relevé à ce fichier CSV")
    robot.add_argument("--une-fois", action="store_true", help="un seul relevé (code 1 si état critique)")
    robot.add_argument("--sans-cameras", action="store_true", help="ne pas ouvrir les caméras")

    entrainement = vues.add_parser("entrainement", help="progression de l'entraînement")
    entrainement.add_argument("--objet", help="objet suivi (défaut : tous)")
    entrainement.add_argument("--suivre", action="store_true", help="actualiser jusqu'à la fin")
    entrainement.add_argument("--intervalle", type=float, default=10.0, help="secondes entre deux lectures")

    bilan = vues.add_parser("bilan", help="taux de réussite par objet et historique des essais")
    bilan.add_argument("--objet", help="limiter à un objet")
    bilan.add_argument("--derniers", type=int, default=10, help="nombre d'essais récents à lister")

    args = parser.parse_args(argv)
    cfg = charger(args)
    vue = {"robot": vue_robot, "entrainement": vue_entrainement, "bilan": vue_bilan}.get(
        args.vue, vue_ensemble
    )
    return vue(cfg, args)


if __name__ == "__main__":
    executer(principal)
