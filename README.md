# IA_test_game : une IA qui apprend à speedrunner le début de Satisfactory

Le vrai jeu est trop lent et trop complexe pour qu'une IA y apprenne par essais-erreurs. Ce projet fournit donc un **mini-Satisfactory** : une simulation simplifiée du Tier 0, de la première action jusqu'au HUB Upgrade 6. Il ne contient ni 3D ni déplacements, seulement l'économie du jeu : ressources, recettes, machines et énergie.

Une partie y dure une fraction de seconde au lieu d'une heure. Une IA peut donc jouer des milliers de parties et apprendre, par renforcement, à finir le Tier 0 le plus vite possible.

## Contenu

| Fichier | Rôle |
|---|---|
| `mini_satisfactory/data.py` | Les données du jeu : objets, recettes, machines, coûts des HUB upgrades |
| `mini_satisfactory/sim.py` | Le simulateur (Python pur, sans dépendance) |
| `mini_satisfactory/env.py` | L'environnement d'entraînement (format Gymnasium) : ce que voit l'IA, ses actions, ses points |
| `mini_satisfactory/bots.py` | Les joueurs de référence : hasard, script simple, script « usine » |
| `train.py` | Entraîne l'IA |
| `play.py` | Compare les joueurs, ou rejoue la partie de l'un d'eux action par action |
| `tests/` | Tests automatiques |

## Installation

Avec Python 3.10 ou plus récent :

```bash
python -m venv .venv
```

Sous Windows :

```bash
.venv\Scripts\activate
```

Sous Linux ou Mac :

```bash
source .venv/bin/activate
```

Puis :

```bash
pip install -r requirements.txt
```

## Utilisation

Comparer les joueurs :

```bash
python play.py
```

Entraîner l'IA (le meilleur modèle est sauvegardé dans `models/ia_tier0.zip`) :

```bash
python train.py
```

Voir la partie de l'IA action par action, une fois entraînée :

```bash
python play.py --joueur ia
```

## Les règles du mini-jeu

- **But** : livrer les 6 HUB upgrades du Tier 0 le plus vite possible (en temps de jeu).
- **Actions du joueur** (chacune prend du temps de jeu) : miner à la main, fabriquer à l'établi, construire une machine, ramasser des feuilles pour le brûleur du HUB, faire la tournée des machines (les vider et les remplir), attendre, livrer le HUB.
- **Machines** : foreuses portatives, Foreuse Mk.1, fonderies, façonneuses. Elles travaillent pendant que le joueur fait autre chose. Fonderies et façonneuses consomment de l'électricité, et le brûleur du HUB a besoin de feuilles.
- **Avant les convoyeurs** (HUB Upgrade 4), il faut remplir et vider les machines à la main. Après, elles sont reliées au stock automatiquement, sauf les foreuses portatives, comme dans le jeu.

Simplifications : pas de carte ni de trajets, un seul stock commun, et quand l'électricité manque, les machines ralentissent au lieu de faire sauter le fusible.

## Comment l'IA apprend

- **Ce qu'elle voit** : le stock, les machines construites, ce qui est débloqué, ce que demande le prochain upgrade, l'électricité et le carburant, les foreuses libres, ce qui attend dans les machines, et le temps écoulé.
- **Ce qu'elle peut faire** : 31 actions. Celles qui sont impossibles sur le moment (pas assez d'objets, pas encore débloqué) sont masquées.
- **Ses points** : −1 par minute de jeu, +10 par HUB upgrade. Ces 10 points sont gagnés progressivement à mesure qu'elle fabrique ce que demande l'upgrade. Sur une partie terminée, le total des upgrades est toujours le même : seul le temps fait la différence.
- **L'algorithme** : PPO avec masque d'actions (`MaskablePPO`, bibliothèque `sb3-contrib`).

## D'où viennent les chiffres

Tout est regroupé dans `mini_satisfactory/data.py`.

- **Repris du jeu (version 1.0)** : coûts et déblocages des HUB upgrades, recettes, coûts des machines, cadence et consommation de la Foreuse Mk.1, consommation des fonderies et façonneuses.
- **Estimations** (marquées `APPROX` dans le fichier) : vitesse de minage et de fabrication à la main, cadence de la foreuse portative, quantité de feuilles ramassées, puissance du brûleur du HUB, gisements autour du départ.

Si tu connais les vraies valeurs, corrige-les dans `data.py` : tout le reste s'adapte.

## Résultats actuels

| Joueur | Résultat |
|---|---|
| Hasard | 1 upgrade en 3 h |
| Script simple (tout à la main + foreuses portatives) | Tier 0 en 75,1 min |
| Script « usine » (fonderies et façonneuses) | Tier 0 en 37,5 min |
| IA | pas encore entraînée |

Le temps du script « usine » est la référence à battre pour l'IA.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

## Prochaines étapes

1. Entraîner l'IA et comparer son temps au script « usine ».
2. Ajouter une fenêtre graphique en Python pour regarder l'IA jouer en direct.
3. Étendre le simulateur aux tiers suivants (Space Elevator, Tier 1 et 2).
4. Plus tard : brancher l'IA sur le vrai jeu, via un mod sur le PC qui lit la partie et exécute les actions.
