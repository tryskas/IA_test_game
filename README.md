# IA_test_game : une IA qui apprend à speedrunner le début de Satisfactory

Le vrai jeu est trop lent et trop complexe pour qu'une IA y apprenne par essais-erreurs. Ce projet fournit donc un **mini-Satisfactory** : une simulation simplifiée du Tier 0, de la première action jusqu'au HUB Upgrade 6. Il ne contient ni 3D ni déplacements, seulement l'économie du jeu : ressources, recettes, machines et énergie.

Une partie y dure une fraction de seconde au lieu d'une heure. Une IA peut donc jouer des milliers de parties et apprendre à finir le Tier 0 le plus vite possible. Une fenêtre Python permet ensuite de la **regarder jouer**, en course contre des bots écrits à la main.

Tout tourne en local, en Python, sans le vrai jeu. Pour brancher l'IA sur le vrai Satisfactory, suis le guide [GUIDE_VRAI_JEU.md](GUIDE_VRAI_JEU.md) (en cours).

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

Entraîner l'IA (environ 15 min, voir plus bas) :

```bash
python train.py
```

Puis la regarder jouer :

```bash
python watch.py
```

> Sous Linux, `pip` installe par défaut la version de PyTorch pour carte graphique (plusieurs Go). Pour une version plus légère, lance d'abord `pip install torch --index-url https://download.pytorch.org/whl/cpu`. Sous Windows et Mac, la version par défaut est déjà légère.

## La fenêtre de jeu (`watch.py`)

Chaque partie est d'abord calculée (en moins d'une seconde), puis jouée à l'écran à la vitesse choisie. La fenêtre montre :

- **HUB** : les 6 upgrades, ce qui manque pour celui en cours, et le temps de chaque upgrade terminé, comparé à celui du script usine (vert : plus rapide, rouge : plus lent) ;
- **Usine** : les gisements et les foreuses posées dessus, les fonderies et constructeurs (en marche, en attente, sortie pleine, sans courant), l'électricité et le carburant ;
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

Entraîner l'IA (environ 15 min sur un PC récent). Le meilleur modèle est enregistré dans `models/ia_tier0.zip` :

```bash
python train.py
```

Entraîner sans l'étape d'imitation, par essais-erreurs uniquement :

```bash
python train.py --sans-imitation
```

Apprendre à l'IA à marcher en terrain accidenté (falaises, ravins, rochers) pour le vrai jeu, environ 30 min. Le meilleur marcheur est enregistré dans `models/marche.zip` (voir le guide) :

```bash
python train_walk.py
```

Regarder l'IA marcher sur ces terrains simulés, vus de dessus (touches 1, 2, 3 pour changer de marcheur, N pour un autre paysage) :

```bash
python watch_walk.py
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
| `watch.py`, `train.py`, `train_walk.py`, `watch_walk.py`, `play.py` | Les commandes ci-dessus |
| `real_game/` | Les scripts pour le vrai jeu : capture d'écran, données du mod, clavier et souris, lecture du texte (voir le guide) |
| `models/ia_tier0.zip` | L'IA entraînée, créée par `train.py` |
| `tests/` | Tests automatiques |

## Les règles du mini-jeu

- **But** : livrer les 6 HUB upgrades du Tier 0 le plus vite possible (en temps de jeu).
- **Actions du joueur** (chacune prend du temps de jeu) : miner à la main, fabriquer à l'établi, construire une machine, ramasser des feuilles pour le brûleur du HUB, faire la tournée des machines (les vider et les remplir), attendre, livrer le HUB.
- **Machines** : atelier d'équipement (pour fabriquer les foreuses portables), foreuses portables, Foreuse Mk.1, fonderies, constructeurs. Elles travaillent pendant que le joueur fait autre chose. Fonderies et constructeurs consomment de l'électricité, et le brûleur du HUB a besoin de feuilles.
- **Avant les convoyeurs** (HUB Upgrade 4), il faut remplir et vider les machines à la main. Après, elles sont reliées au stock automatiquement, sauf les foreuses portables, comme dans le jeu.

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

- **Lu dans le jeu** (version 1.2.4.1, avec le mod Ficsit Remote Monitoring) : coûts et déblocages des HUB upgrades, recettes avec leur durée à la main et en machine, coûts des machines et de l'atelier d'équipement. Un test vérifie que les coûts des upgrades restent identiques à ceux du jeu.
- **Repris de la documentation du jeu** : cadence et consommation de la Foreuse Mk.1, consommation des fonderies et constructeurs.
- **Estimations** (marquées `APPROX` dans le fichier) : vitesse de minage à la main, cadence de la foreuse portable, quantité de feuilles ramassées, puissance du brûleur du HUB, gisements autour du départ.

Si tu connais les vraies valeurs, corrige-les dans `data.py`, puis réentraîne l'IA : le reste s'adapte.

## Résultats

| Joueur | Résultat |
|---|---|
| Script usine (fonderies et constructeurs) | Tier 0 en 45,6 min |
| Script simple (tout à la main + foreuses portables) | Tier 0 en 121,9 min |
| Hasard | 1 upgrade en 3 h |
| IA | à entraîner sur ton PC avec `python train.py` |

Avec les durées réelles de fabrication à la main (6 s par lingot), construire des machines rapporte encore plus qu'avec les estimations de départ.

Avec les estimations de départ, l'IA entraînée (imitation, puis 500 000 actions seule) avait fini le Tier 0 en 36,6 min, contre 37,5 min pour le script usine. Sans imitation, elle n'avait pas dépassé 5 upgrades sur 6 en 3 h.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

## Prochaines étapes

1. Étendre le simulateur aux tiers suivants (Space Elevator, Tier 1 et 2).
2. Varier la carte de départ (nombre et pureté des gisements) pour que l'IA s'adapte au lieu d'apprendre une seule route.
3. Brancher l'IA sur le vrai jeu : en cours, voir [GUIDE_VRAI_JEU.md](GUIDE_VRAI_JEU.md).
