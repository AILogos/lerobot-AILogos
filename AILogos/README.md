# AILogos — apprendre au robot à attraper des objets, puis le commander à la voix

Ce dossier contient les programmes pour **enregistrer, entraîner, tester et monitorer** un bras
SO-101 qui attrape des objets connus (une balle, un crayon…), puis une **V2** qui traduit des
commandes vocales comme « attrape la balle » en actions du robot.

Tout repose sur LeRobot : les programmes préparent et lancent `lerobot-record`, `lerobot-train` et le
moteur de `lerobot-rollout` à partir d'un seul fichier, [`config.yaml`](./config.yaml).

| Étape | Programme               | Rôle                                                                |
| ----- | ----------------------- | ------------------------------------------------------------------- |
| 1     | `enregistrer.py`        | Enregistrer des démonstrations de saisie en téléopération           |
| 2     | `entrainer.py`          | Entraîner une politique ACT par objet                               |
| 3     | `tester.py`             | Faire des essais notés sur le robot et mesurer le taux de réussite  |
| 4     | `monitorer.py`          | Suivre le robot (moteurs, caméras), l'entraînement et les résultats |
| V2    | `v2_commande_vocale.py` | Micro → Whisper local → ordre compris → saisie de l'objet demandé   |

> **Ce que « attraper » veut dire ici** : saisir un objet **posé** dans la zone de travail. Attraper
> un objet lancé en vol demande une tout autre réactivité que celle d'un SO-101 piloté à 30 Hz.

## Matériel prévu

- Bras suiveur **SO-101** + bras **leader SO-101** (téléopération pour l'enregistrement).
- **2 caméras** USB : une fixe face à la scène (`front`), une sur le poignet (`wrist`).
- Un PC avec GPU (< 8 Go suffit pour ACT) pour l'entraînement, un micro pour la V2.

Un autre robot supporté par LeRobot fonctionne aussi : il suffit d'adapter les sections `robot` et
`teleop` de `config.yaml` (mêmes champs que les options `--robot.*` / `--teleop.*` de LeRobot).

## Installation

Depuis la racine du dépôt :

```bash
uv sync --locked --extra core_scripts --extra feetech --extra training
```

Pour la commande vocale (V2) :

```bash
uv pip install -r AILogos/requirements-voix.txt
```

Sous Linux, le micro demande aussi PortAudio : `sudo apt install libportaudio2`.

## Réglage initial (une seule fois)

1. Assembler, configurer et calibrer les bras en suivant [`AGENT_GUIDE.md`](../AGENT_GUIDE.md) §4.1 à
   §4.5 (`lerobot-find-port`, `lerobot-setup-motors`, `lerobot-calibrate`, `lerobot-teleoperate`).
2. Trouver les caméras : `uv run lerobot-find-cameras opencv`.
3. Reporter dans [`config.yaml`](./config.yaml) : les ports (`robot.port`, `teleop.port`), les `id`
   utilisés à la calibration, et les index des caméras.
4. Vérifier que tout répond :

```bash
uv run python -m AILogos.monitorer robot
```

## Les 4 étapes

Toutes les commandes se lancent **depuis la racine du dépôt**. La vue d'ensemble indique à tout
moment où en est chaque objet et quelle commande lancer ensuite :

```bash
uv run python -m AILogos.monitorer
```

### 1. Enregistrer des démonstrations

```bash
uv run python -m AILogos.enregistrer --objet balle
```

Vous pilotez le suiveur avec le leader ; chaque épisode montre une saisie complète
(`→` épisode suivant, `←` refaire, `Échap` terminer). 50 épisodes pour commencer, avec l'objet
toujours au même endroit, puis varier progressivement. Répéter pour chaque objet :

```bash
uv run python -m AILogos.enregistrer --objet crayon
```

`--reprendre` ajoute des épisodes à un dataset existant, `--episodes 30` change leur nombre.

### 2. Entraîner

```bash
uv run python -m AILogos.entrainer --objet balle
```

Une politique **ACT** est entraînée par objet (`--tous` pour les enchaîner). Le nombre de pas est
calculé à partir de la taille du dataset (10 époques par défaut) ; une partie des épisodes est
gardée de côté pour mesurer une perte de validation. Suivre la progression depuis un autre terminal :

```bash
uv run python -m AILogos.monitorer entrainement --objet balle --suivre
```

Autres cas : `--reprendre` après une coupure, `--device mps` sur Mac, et tout ce qui suit `--` est
transmis à `lerobot-train` — par exemple `-- --job.target=a10g-small` pour entraîner sur un GPU
distant (Hugging Face Jobs, avec `envoyer_sur_hub: true`).

### 3. Tester

```bash
uv run python -m AILogos.tester --objet balle
```

Pour chaque essai : placer l'objet, `Entrée` pour lancer, `Entrée` (ou `Ctrl+C`) dès que l'objet
est attrapé ou l'échec évident, puis noter l'essai (`o` réussi / `n` raté). Le bras revient à sa
position de départ et se relâche après chaque essai. Options utiles :

- `--afficher` : caméras et actions en direct dans Rerun ;
- `--simulation` : aucun matériel, la politique tourne sur un robot simulé (vérifie qu'elle se
  charge et tient la cadence avant d'alimenter le bras).

### 4. Monitorer

```bash
uv run python -m AILogos.monitorer bilan
```

| Vue            | Ce qu'elle montre                                                                      |
| -------------- | -------------------------------------------------------------------------------------- |
| _(sans rien)_  | Pour chaque objet : données, politique, résultats, prochaine commande à lancer         |
| `robot`        | Température, tension, charge et courant de chaque moteur, état des caméras, en direct  |
| `entrainement` | Avancement, courbe de perte, perte de validation, temps restant                        |
| `bilan`        | Taux de réussite par objet et historique des essais (tests notés et commandes vocales) |

`monitorer robot` ouvre le bras en lecture seule ; `--csv releve.csv` conserve les relevés. Les
seuils d'alerte se règlent dans la section `sante` de `config.yaml`. Avant et après chaque saisie,
`tester.py` et la commande vocale relèvent aussi l'état des moteurs et refusent de lancer une saisie
si un moteur est en défaut (surchauffe, surcharge).

## V2 — Commande vocale

```bash
uv run python -m AILogos.v2_commande_vocale
```

Dites « **attrape la balle** » : le robot annonce ce qu'il a compris, exécute la politique de
l'objet demandé, puis revient au repos.

| Vous dites                                         | Le robot                                                |
| -------------------------------------------------- | ------------------------------------------------------- |
| « attrape la balle », « prends le stylo »…         | lance la saisie de l'objet (les synonymes sont compris) |
| « attrape la balle puis le crayon »                | enchaîne les deux saisies                               |
| « stop », « arrête »                               | interrompt la saisie et revient au repos                |
| « attrape la tasse » (objet inconnu)               | répond qu'il ne connaît pas cet objet                   |
| « n'attrape pas la balle », conversation ordinaire | ne fait rien                                            |
| « au revoir »                                      | quitte le programme                                     |

Pour mettre au point sans micro ni robot :

```bash
uv run python -m AILogos.v2_commande_vocale --mode clavier --simulation
```

Autres options : `--texte "attrape la balle"` (un seul ordre puis quitte), `--lister-micros`,
`--afficher` (Rerun), `--sans-retour-vocal`.

Comment ça marche :

```text
micro ──► détection de parole ──► Whisper (local) ──► interpréteur ──► politique ACT de l'objet ──► robot
          voix/ecoute.py          voix/ecoute.py      voix/interpreteur.py     moteur.py
```

- La transcription tourne **en local** (faster-whisper) : aucun son ne quitte l'ordinateur. Le
  modèle est téléchargé au premier lancement, puis reste en cache.
- Un ordre de saisie exige un **verbe de saisie et un objet connu** ; un objet cité dans une
  conversation ne déclenche rien. Dans une pièce où l'on discute, renseigner `voix.mot_reveil`
  (ex. `robot` → « robot, attrape la balle »).
- Le micro est coupé pendant que le robot parle, pour qu'il ne s'entende pas lui-même.

### Sécurité

- **La voix n'est pas un arrêt d'urgence** : il faut 1 à 2 secondes pour comprendre « stop ».
  `Ctrl+C` interrompt immédiatement la saisie ; garder l'alimentation du bras à portée de main.
- Le bras doit être **en position de repos** avant chaque ordre : il est relâché entre deux saisies.
- Pour les premiers essais d'une nouvelle politique, limiter les déplacements avec
  `robot.max_relative_target: 10` dans `config.yaml`.

## Ajouter un objet

1. L'ajouter dans `config.yaml` :

```yaml
objets:
  cube:
    tache: "Attrape le cube"
    synonymes: [cube, dé, bloc]
```

2. `enregistrer --objet cube` → `entrainer --objet cube` → `tester --objet cube`. La commande vocale
   le reconnaît aussitôt.

## Organisation du dossier

| Fichier                  | Contenu                                                                 |
| ------------------------ | ----------------------------------------------------------------------- |
| `config.yaml`            | Tous les réglages : matériel, objets, entraînement, seuils, voix        |
| `configuration.py`       | Lecture et validation de `config.yaml`, noms des datasets/politiques    |
| `commandes.py`           | Construction des commandes `lerobot-record` / `lerobot-train`           |
| `moteur.py`              | Exécute la politique d'un objet sur le robot (moteur `lerobot.rollout`) |
| `sante.py`, `journal.py` | Télémétrie des moteurs ; journal des essais                             |
| `robot_simule.py`        | Robot simulé pour `--simulation` et les tests                           |
| `voix/`                  | Détection de parole, transcription, interprétation, retour vocal        |
| `tests/`                 | Tests automatiques, sans matériel                                       |

Les sorties vont dans `outputs/` (ignoré par git) : politiques dans `outputs/train/ailogos_act_<objet>/`,
journaux et essais dans `outputs/ailogos/`. Les datasets sont dans le cache LeRobot
(`~/.cache/huggingface/lerobot/local/ailogos_attrape_<objet>`).

## Tests

```bash
uv run pytest AILogos/tests -q
```

Les tests n'utilisent ni robot ni micro : la boucle de contrôle de LeRobot est exercée sur le robot
simulé avec une petite politique ACT, et la partie vocale avec des signaux synthétiques.

## Dépannage

| Symptôme                                 | Piste                                                                                    |
| ---------------------------------------- | ---------------------------------------------------------------------------------------- |
| Erreur de communication avec les moteurs | Câblage et alimentation d'abord (voir `AGENT_GUIDE.md` §4.5), puis `monitorer robot`     |
| La politique ignore l'objet              | Cadrage ou éclairage des caméras ; les caméras ont-elles bougé depuis l'enregistrement ? |
| Taux de réussite faible                  | 10 à 20 démonstrations ciblées sur les échecs (`enregistrer --reprendre`), réentraîner   |
| « Visual feature mismatch »              | Les noms des caméras de `config.yaml` ont changé depuis l'enregistrement                 |
| Le robot réagit à la conversation        | Renseigner `voix.mot_reveil` ; augmenter `voix.seuil_parole`                             |
| Les ordres ne sont pas compris           | `--lister-micros` puis `voix.micro` ; essayer `voix.modele: medium`                      |
| La transcription ralentit la saisie      | `voix.modele: base`, ou `voix.device: cuda` si la mémoire GPU le permet                  |

## Pour aller plus loin

Avec un GPU de 12 Go ou plus, une politique conditionnée par le langage (SmolVLA) peut remplacer
les politiques ACT par objet : une seule politique reçoit la phrase « attrape la balle » comme
consigne. La structure reste la même — `objets.<nom>.politique` permet déjà de faire pointer
plusieurs objets vers une politique commune, et `tache` est transmise à la politique.
