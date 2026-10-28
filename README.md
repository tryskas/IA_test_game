# IA_test_game : une IA qui apprend à speedrunner le début de Satisfactory

Le vrai jeu est trop lent et trop complexe pour qu'une IA y apprenne par essais-erreurs. Ce projet fournit donc un **mini-Satisfactory** : une simulation simplifiée du Tier 0, de la première action jusqu'au HUB Upgrade 6. Il ne contient ni 3D ni déplacements, seulement l'économie du jeu : ressources, recettes, machines et énergie.

Une partie y dure une fraction de seconde au lieu d'une heure. Une IA peut donc jouer des milliers de parties et apprendre à finir le Tier 0 le plus vite possible. Une fenêtre Python permet ensuite de la **regarder jouer**, en course contre des bots écrits à la main.

Tout tourne en local, en Python. Le vrai jeu n'est pas utilisé.

## Démarrage rapide

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

Regarder l'IA jouer (une IA déjà entraînée est fournie dans `models/ia_tier0.zip`) :

```bash
python watch.py
```

> Sous Linux, `pip` installe par défaut la version de PyTorch pour carte graphique (plusieurs Go). Pour une version plus légère, lance d'abord `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Sous Windows et Mac, la version par défaut est déjà légère.

## La fenêtre de jeu (`watch.py`)

Chaque partie est d'abord calculée (en moins d'une seconde), puis jouée à l'écran à la vitesse choisie. La fenêtre montre :

- **HUB** : les 6 upgrades, ce qui manque pour celui en cours, et le temps de chaque upgrade terminé, comparé à celui du script usine (vert : plus rapide, rouge : plus lent) ;
- **Usine** : les gisements et les foreuses posées dessus, les fonderies et façonneuses (en marche, en attente, sortie pleine, sans courant), l'électricité et le carburant ;
- **Pionnier** : l'action en cours, le stock et le journal des dernières actions ;
- **La course** : où en est chaque joueur au même instant de jeu.

| Touche | Effet |
|---|---|
| Espace | pause / reprendre |
| ← → | reculer / avancer de 30 s |
| ↑ ↓ | changer la vitesse (×1 à ×300) |
| 1 à 4 (ou clic dans « La course ») | changer de joueur suivi |
| Début | revenir au début |
| Échap | quitter |

Un clic sur la ligne de temps saute à ce moment. Options utiles : `python watch.py --vitesse 30` ou `python watch.py --suivre usine`.

## Les autres commandes

Entraîner l'IA (environ 15 min sur un PC récent). Le meilleur modèle remplace `models/ia_tier0.zip` :

```bash
python train.py
```

Entraîner sans l'étape d'imitation, par essais-erreurs uniquement :

```bash
python train.py --sans-imitation
```

Comparer les joueurs dans le terminal :

```bash
python play.py
```

Lister toutes les actions de la partie d'un joueur :

```bash
python play.py --joueur ia
```

## Contenu

| Fichier | Rôle |
|---|---|
| `mini_satisfactory/data.py` | Les données du jeu : objets, recettes, machines, coûts des HUB upgrades |
| `mini_satisfactory/sim.py` | Le simulateur (Python pur, sans dépendance) |
| `mini_satisfactory/env.py` | L'environnement d'entraînement (format Gymnasium) : ce que voit l'IA, ses actions, ses points |
| `mini_satisfactory/bots.py` | Les joueurs de référence : hasard, script simple, script usine |
| `mini_satisfactory/ai_player.py` | L'IA entraînée, branchée comme un joueur |
| `mini_satisfactory/replay.py` | Enregistre une partie, un instantané toutes les 2 s de jeu |
| `mini_satisfactory/viewer.py` | La fenêtre pygame |
| `watch.py`, `train.py`, `play.py` | Les commandes ci-dessus |
| `models/ia_tier0.zip` | L'IA entraînée |
| `tests/` | Tests automatiques |

## Les règles du mini-jeu

- **But** : livrer les 6 HUB upgrades du Tier 0 le plus vite possible (en temps de jeu).
- **Actions du joueur** (chacune prend du temps de jeu) : miner à la main, fabriquer à l'établi, construire une machine, ramasser des feuilles pour le brûleur du HUB, faire la tournée des machines (les vider et les remplir), attendre, livrer le HUB.
- **Machines** : foreuses portatives, Foreuse Mk.1, fonderies, façonneuses. Elles travaillent pendant que le joueur fait autre chose. Fonderies et façonneuses consomment de l'électricité, et le brûleur du HUB a besoin de feuilles.
- **Avant les convoyeurs** (HUB Upgrade 4), il faut remplir et vider les machines à la main. Après, elles sont reliées au stock automatiquement, sauf les foreuses portatives, comme dans le jeu.

Simplifications : pas de carte ni de trajets, un seul stock commun, et quand l'électricité manque, les machines ralentissent au lieu de faire sauter le fusible.

## Comment l'IA apprend

L'entraînement se fait en deux étapes :

1. **Imitation** : l'IA regarde le script usine jouer 60 parties, dont certaines avec des détours au hasard pour qu'elle voie aussi comment s'en sortir. Elle apprend à choisir la même action que lui.
2. **Entraînement seule** (apprentissage par renforcement, algorithme PPO avec masque d'actions) : elle joue des milliers de parties et garde ce qui la rend plus rapide.

Sans l'étape 1, l'IA apprend par pur essais-erreurs. Elle finit par faire tout à la main, sans jamais construire de machines : une machine coûte des objets tout de suite et ne rapporte que plus tard, ce qu'elle ne découvre pas seule assez vite.

- **Ce qu'elle voit** : le stock, les machines construites, ce qui est débloqué, ce que demandent le prochain upgrade et tout le reste du Tier 0, l'électricité et le carburant, les gisements libres, ce qui attend dans les machines, et le temps écoulé.
- **Ce qu'elle peut faire** : 31 actions. Celles qui sont impossibles sur le moment (pas assez d'objets, pas encore débloqué) sont masquées.
- **Ses points** : −1 par minute de jeu. Chaque HUB upgrade vaut 15 points : 10 gagnés au fur et à mesure qu'elle fabrique ce qu'il demande, même à l'avance, et 5 à la livraison. Sur une partie terminée, ces 90 points sont toujours les mêmes : seul le temps fait la différence.

## D'où viennent les chiffres

Tout est regroupé dans `mini_satisfactory/data.py`.

- **Repris du jeu (version 1.0)** : coûts et déblocages des HUB upgrades, recettes, coûts des machines, cadence et consommation de la Foreuse Mk.1, consommation des fonderies et façonneuses.
- **Estimations** (marquées `APPROX` dans le fichier) : vitesse de minage et de fabrication à la main, cadence de la foreuse portative, quantité de feuilles ramassées, puissance du brûleur du HUB, gisements autour du départ.

Si tu connais les vraies valeurs, corrige-les dans `data.py`, puis réentraîne l'IA : le reste s'adapte.

## Résultats

| Joueur | Résultat |
|---|---|
| **IA** (imitation, puis 500 000 actions d'entraînement seule) | **Tier 0 en 36,6 min** |
| Script usine (fonderies et façonneuses) | Tier 0 en 37,5 min |
| Script simple (tout à la main + foreuses portatives) | Tier 0 en 75,1 min |
| IA sans imitation (600 000 actions) | 5 upgrades sur 6 au mieux, en 3 h |
| Hasard | 1 upgrade en 3 h |

L'IA a appris la stratégie du script usine, puis l'a un peu améliorée : elle gagne environ 1 min, surtout sur les HUB Upgrades 2 et 3.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

## Prochaines étapes

1. Étendre le simulateur aux tiers suivants (Space Elevator, Tier 1 et 2).
2. Varier la carte de départ (nombre et pureté des gisements) pour que l'IA s'adapte au lieu d'apprendre une seule route.
3. Plus tard : brancher l'IA sur le vrai jeu, via un mod sur le PC qui lit la partie et exécute les actions.
