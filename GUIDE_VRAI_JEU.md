# Guide : brancher l'IA sur le vrai Satisfactory

Pour jouer au vrai jeu, l'IA a besoin de quatre choses :

1. **des yeux** : voir l'écran du jeu (capture d'écran en continu) ;
2. **des données exactes** : l'inventaire, les machines, l'électricité, sans les deviner à l'image (grâce à un mod) ;
3. **des mains** : appuyer sur les touches et bouger la souris ;
4. **un traducteur** : transformer une action du mini-jeu (« fabriquer des barres de fer ») en gestes dans le vrai jeu.

Ce guide installe et vérifie ces pièces une par une. Chaque étape a son script de vérification dans le dossier `real_game/`. **Après chaque étape, envoie-moi ce que le script affiche** (et les fichiers qu'il crée) : on construira l'étape suivante à partir de tes vraies données.

À savoir avant de commencer :

- Je ne peux pas lancer le jeu de mon côté. Ces scripts ont été écrits sans le jeu : tes retours servent à les ajuster.
- Le mini-jeu ignore les déplacements et la construction en 3D. Sur le vrai jeu, ce sont les parties les plus difficiles. On commencera par ce qui se fait sans bouger, comme fabriquer à l'établi.
- Les mods et l'automatisation sont interdits dans les classements de speedrun : c'est une expérience, pas une tentative de record.
- Utilise une **sauvegarde dédiée**, pas ta partie principale.

## Étape 0 : préparer le PC et le jeu

### Installer les outils

1. Installe **Python 3.12** depuis [python.org](https://www.python.org/downloads/windows/). Pendant l'installation, coche **« Add python.exe to PATH »**.
2. Installe **Git** depuis [git-scm.com](https://git-scm.com/download/win) (ou utilise GitHub Desktop).
3. Ouvre **PowerShell** (touche Windows, tape `powershell`) et récupère le projet :

```bash
git clone https://github.com/tryskas/IA_test_game.git
```

```bash
cd IA_test_game
```

```bash
git checkout claude/happy-hawking-zw1qmr
```

4. Crée l'environnement Python et installe les bibliothèques (quelques minutes) :

```bash
python -m venv .venv
```

Par défaut, PowerShell refuse de lancer le script qui active l'environnement (erreur « l'exécution de scripts est désactivée sur ce système »). Autorise-le une fois pour toutes, pour ton compte uniquement :

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Réponds « O » (Oui) si PowerShell demande confirmation. Ce réglage autorise les scripts créés sur ton PC, comme celui de l'environnement Python. Les scripts téléchargés d'Internet doivent toujours être signés.

Puis active l'environnement et installe les bibliothèques :

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt -r real_game/requirements.txt
```

Quand l'environnement est actif, `(.venv)` apparaît au début de la ligne. À chaque nouvelle fenêtre PowerShell, relance `.venv\Scripts\activate` dans le dossier du projet.

Ce que ces bibliothèques font :

| Bibliothèque | Rôle |
|---|---|
| `bettercam` | capture d'écran rapide via DirectX (plus de 100 images/s) |
| `mss` | capture d'écran plus lente, en secours |
| `opencv` | traitement d'image : découper, comparer, chercher un motif |
| `easyocr` | lire le texte affiché à l'écran (en français et en anglais) |
| `pygetwindow` | trouver la fenêtre du jeu |
| `pynput` | surveiller la touche d'arrêt d'urgence (F12) |

### Régler le jeu

Dans les options du jeu :

- **affichage** : mode fenêtré sans bordure (borderless), à la résolution de ton écran (par exemple 2560 x 1440). Garde ensuite toujours la même : les positions des menus à l'écran en dépendent ;
- **flou de mouvement** : désactivé (les captures seront plus nettes) ;
- **taille de l'interface** : laisse la valeur par défaut, et ne la change plus ensuite ;
- **langue** : français, et garde-la (la lecture du texte en dépend).

Les touches sont réglées pour un clavier AZERTY (Z, Q, S, D), comme le jeu les affiche. Si tu les as changées dans Options > Commandes, reporte les lettres affichées par le jeu dans `real_game/config.py`, section `KEYS`. Le clavier et la souris sont pilotés par `real_game/inputs.py`, qui envoie la touche physique correspondant à chaque lettre selon ta disposition de clavier.

Enfin, crée une **nouvelle partie** dédiée aux essais, dans la zone de départ la plus simple (Grass Fields).

## Étape 1 : les yeux (capture d'écran)

Lance le jeu et charge ta partie. Dans le terminal :

```bash
python real_game/check_capture.py
```

Pendant 5 secondes, bouge un peu la caméra dans le jeu. Le script affiche :

- la fenêtre trouvée et sa taille ;
- la méthode de capture et le nombre d'images par seconde (30 ou plus, c'est bien) ;
- 3 captures enregistrées dans le dossier `captures/`.

**Envoie-moi** ce que le script a affiché et une des captures (glisse l'image dans la conversation).

En cas de problème :

- « Je ne trouve pas la fenêtre du jeu » : le jeu doit être lancé, avec une partie chargée.
- Capture noire : passe le jeu en mode fenêtré sans bordure.

## Étape 2 : les données exactes (mod Ficsit Remote Monitoring)

Lire l'inventaire ou l'électricité à l'image est fragile : un chiffre mal lu, et l'IA se trompe. Le mod **Ficsit Remote Monitoring** fait tourner un petit serveur dans le jeu, qui donne ces données exactes à Python. Il ne modifie pas la partie : il se contente de lire.

1. Installe **Satisfactory Mod Manager** depuis [ficsit.app](https://ficsit.app).
2. Dans le Mod Manager, cherche **Ficsit Remote Monitoring** et installe-le. Ses dépendances s'installent toutes seules.
3. Vérifie que le Mod Manager indique le mod comme compatible avec ta version du jeu (affichée en haut à gauche de l'écran, par exemple 1.2.4.1).
4. Lance le jeu depuis le Mod Manager et charge ta partie.
5. Ouvre le chat du jeu (touche Entrée) et tape :

```text
/frm http start
```

Pour ne plus avoir à le taper, active le démarrage automatique du serveur web dans les options du mod (menu principal du jeu, Options, section du mod).

6. Dans le terminal :

```bash
python real_game/check_game_data.py
```

Le script interroge le mod sur chaque type de données (joueur, inventaire, électricité, machines, jalons...), dit lesquels répondent et les enregistre dans `captures/frm/`.

**Envoie-moi** ce que le script a affiché, et les fichiers `getPlayer.json` et `getSchematics.json` s'ils existent. Je m'en servirai pour traduire l'état du vrai jeu dans le format du mini-jeu.

Si Windows demande d'autoriser le jeu sur le réseau, accepte pour les réseaux privés.

## Le copilote : l'IA te conseille pendant que tu joues

Dès que l'étape 2 marche, l'IA peut lire ta partie et te dire quoi faire. Tu joues toi-même, elle conseille. Il faut d'abord l'entraîner, une seule fois (environ 15 min) :

```bash
python train.py
```

Puis, le jeu lancé avec le mod et une partie chargée :

```bash
python real_game/copilot.py
```

Toutes les 2 secondes, le copilote lit ton inventaire et tes HUB upgrades, et affiche le prochain upgrade avec ce qui manque, ainsi que l'action conseillée par l'IA. Pour miner ou poser une foreuse, il indique aussi le gisement libre le plus proche : sa distance et de combien tourner (vers la droite ou la gauche).

Limites de cette première version :

- il ne voit pas encore tes machines, seulement l'atelier d'équipement ;
- le sens « droite / gauche » vient d'un calcul pas encore vérifié dans le jeu : dis-moi s'il est inversé ;
- le mini-jeu commence une fois le HUB posé : termine d'abord le début du tutoriel à la main.

## Étape 3 : les mains (clavier et souris)

Place ton personnage dans un endroit sûr : sur un sol plat, rien en main, loin d'un précipice. Puis :

```bash
python real_game/check_controls.py
```

Tu as 5 secondes pour cliquer dans le jeu. Le personnage doit ensuite :

1. tourner la caméra à droite, puis à gauche ;
2. avancer, puis reculer ;
3. sauter ;
4. ouvrir l'inventaire, puis le fermer.

Sécurités : **F12 arrête tout immédiatement**, et le script s'arrête aussi dès que le jeu n'est plus la fenêtre active (par exemple avec Alt+Tab).

En cas de problème :

- Le personnage tourne mais n'avance pas : la touche d'avance ne correspond pas à celle du jeu. Vérifie dans Options > Commandes et corrige `forward` dans `real_game/config.py` (pareil pour les autres touches).
- Rien ne bouge du tout : relance le terminal en tant qu'administrateur.

**Envoie-moi** la liste des mouvements qui ont marché et de ceux qui n'ont pas marché.

## Étape 4 : lire l'écran (OCR)

Certaines informations ne sont pas dans le mod, ou se lisent plus simplement à l'écran (un menu ouvert, une recette sélectionnée). Pour tester la lecture de texte :

1. Dans le jeu, ouvre le **terminal du HUB** ou l'**établi**.
2. Relance `python real_game/check_capture.py` pour avoir une capture de ce menu.
3. Lance la lecture sur cette capture (remplace le nom du fichier par le tien) :

```bash
python real_game/check_ocr.py captures/capture_20261028_120000_1.png
```

La première fois, le script télécharge les modèles de lecture (environ 100 Mo). Il affiche chaque texte trouvé avec sa position, et crée une copie de l'image où chaque texte est encadré (fichier `..._ocr.png`). Les cadres verts sont sûrs, les orange moins.

**Envoie-moi** l'image annotée.

## Étape 5 : l'IA joue

Tout passe par une seule commande, `real_game/agent.py`. Chaque commande laisse 5 secondes pour cliquer dans le jeu. **F12 arrête tout**, et quitter la fenêtre du jeu aussi. Le jeu doit tourner avec le mod, avec `/frm http start` tapé dans le chat.

1. **Calibrer**, une seule fois : place-toi sur un terrain plat avec une quinzaine de mètres dégagés devant toi. Le personnage tourne un peu, avance deux fois, et le résultat est enregistré dans `real_game/calibration.json`.

```bash
python real_game/agent.py calibrer
```

2. **Se déplacer** : le personnage marche jusqu'au gisement de fer libre le plus proche.

```bash
python real_game/agent.py aller fer
```

3. **Miner à la main** : il y va, cherche le gisement du regard, puis mine 10 minerais.

```bash
python real_game/agent.py miner fer 10
```

4. **Les menus**, à tester un par un. L'IA lit le texte à l'écran pour trouver les boutons :

```bash
python real_game/agent.py equiper
```

```bash
python real_game/agent.py construire hub
```

```bash
python real_game/agent.py construire etabli
```

```bash
python real_game/agent.py fabriquer "Lingot de fer" 5
```

L'IA retient où elle a construit l'établi et l'atelier (fichier `real_game/memory.json`, propre à chaque partie), pour y revenir ensuite.

5. **Le terminal du HUB** : choisir le prochain jalon, puis y déposer ce que tu portes.

```bash
python real_game/agent.py jalon
```

```bash
python real_game/agent.py deposer
```

6. **Jouer** : l'IA fait le début de partie si besoin (équiper, HUB, établi), puis remplit les jalons l'un après l'autre. Elle choisit le jalon le moins coûteux, mine et fabrique à l'établi ce qui manque en remontant la chaîne (béton à partir du calcaire, fil à partir du cuivre...), puis va tout déposer au terminal. Son plan est calculé à partir des données du jeu : coûts restants, recettes débloquées. Pour l'instant, tout est fait à la main : les machines viendront ensuite.

```bash
python real_game/agent.py jouer
```

## Étape 6 : la vision

L'IA analyse l'image du jeu en continu, quelques fois par seconde :

- **créatures** : un détecteur d'objets à vocabulaire ouvert (YOLO-World) les trouve à l'écran à partir de mots (« créature », « animal », « araignée »...), sans entraînement sur Satisfactory. L'IA sait ainsi de combien se tourner pour faire face à une créature et la frapper avec le repousse-xénos. Si sa vie descend trop bas, elle fuit vers le HUB ;
- **obstacles** : un modèle de profondeur (Depth Anything V2) estime la distance de chaque point de l'image. Chaque direction devant elle est classée dégagée ou bloquée, et elle contourne ce qui est dans le chemin ;
- **secours** : si sa vie baisse sans qu'elle ait rien vu, elle s'arrête, se retourne pour chercher l'attaquant, ou fuit. Après une mort, elle clique sur le bouton de réapparition et reprend son plan. Elle retient les endroits où elle a été attaquée et évite ensuite les gisements à moins de 50 m.

**Carte apprise et chemins** : ce que l'IA voit dans le chemin (rochers, falaises, à-pics), les endroits où elle reste bloquée ou tombe, et ceux où elle est attaquée sont notés sur une carte du terrain, gardée dans `real_game/memory.json`. Pour aller quelque part, elle calcule un chemin sur cette carte (algorithme A*) qui contourne les obstacles connus et s'écarte des zones dangereuses. Elle le recalcule dès qu'elle découvre un nouvel obstacle. Après deux attaques sur le chemin d'un même gisement, ou si elle n'arrive pas à l'atteindre, elle en choisit un autre.

**Gestes naturels** : la caméra tourne progressivement (elle accélère puis ralentit), la trajectoire est corrigée en marchant, l'inclinaison du regard est suivie et changée en douceur, et le curseur glisse dans les menus.

**Combat** : l'IA se retourne d'abord (les attaques viennent souvent de derrière), fait face et frappe. Elle ne fuit que si sa vie passe sous 25 % pendant qu'elle est encore touchée. Une chute n'est pas prise pour une attaque. Après un combat, si sa vie est sous 60 %, elle rentre au HUB et attend qu'elle remonte.

Vérifie d'abord le **champ de vision** du jeu (Options > Vidéo) et reporte-le dans `real_game/config.py` (`FOV`, 90 par défaut). Il sert à transformer la position d'une créature à l'écran en angle.

Pour voir ce que l'IA voit, sans qu'elle touche aux commandes, lance :

```bash
python real_game/agent.py voir
```

Pendant 20 secondes, elle enregistre dans `captures/vision/` des images annotées : créatures encadrées en orange avec leur angle, directions dégagées en vert et bloquées en rouge. Le premier lancement télécharge les modèles (environ 500 Mo). Ajoute `--sans-vision` à une commande pour jouer sans analyse d'image.

## Étape 7 : montrer une partie à l'IA

Pour que l'IA joue seule du début à la fin, elle doit connaître chaque menu du jeu (inventaire, capsule, construction du HUB, établi, terminal). Joue une fois le début d'une **nouvelle partie** pendant que ce script enregistre :

```bash
python real_game/record_demo.py
```

Il garde une image à chaque clic, les touches utilisées et leur durée, une petite image toutes les 5 s, et l'état du jeu via le mod. Rien n'est enregistré quand une autre fenêtre est active. F12 arrête l'enregistrement, qui va dans `demos/`.

Ensuite, envoie la démonstration sur la branche pour que je l'analyse :

```bash
git pull
```

```bash
git add demos
```

```bash
git commit -m "Démonstration du début de partie"
```

```bash
git push
```

## Ensuite : relier l'IA au vrai jeu

Une fois les quatre étapes validées, l'IA jouera en boucle :

1. **lire** l'état du jeu (le mod pour les chiffres, l'écran pour les menus) ;
2. le **traduire** dans le format du mini-jeu (stock, machines, upgrades livrés) ;
3. **choisir** une action : c'est la même IA que dans `watch.py` ;
4. l'**exécuter** avec une suite de gestes enregistrée pour cette action (une « macro ») ;
5. **vérifier** dans le mod que l'action a marché, puis recommencer.

Le mini-jeu commence une fois le HUB posé. Le début du tutoriel du vrai jeu (équiper le repousse-xénos, démonter la capsule d'atterrissage, construire le HUB) se fera d'abord à la main, puis avec une macro dédiée.

Les macros sont à construire une par une, de la plus simple à la plus difficile :

| Action du mini-jeu | Gestes dans le vrai jeu | Difficulté |
|---|---|---|
| Fabriquer à la main | ouvrir l'établi, cliquer la recette, maintenir le bouton de fabrication | facile : écran fixe |
| Livrer le HUB Upgrade | ouvrir le terminal du HUB, déposer les objets | facile |
| Miner à la main | se placer face au gisement, maintenir la touche d'interaction | moyenne : il faut se déplacer |
| Ramasser des feuilles | trouver des plantes et les ramasser | moyenne |
| Faire la tournée des machines | aller à chaque machine, l'ouvrir, la vider ou la remplir | difficile |
| Construire une machine | ouvrir le menu de construction, viser l'emplacement, poser | difficile : placement en 3D |

Pour les déplacements, on enregistrera la position de chaque lieu utile (HUB, gisements, machines) grâce au mod. Le personnage s'y rendra en orientant la caméra vers la cible, puis en avançant.

## Récapitulatif : ce que tu m'envoies

| Étape | À m'envoyer |
|---|---|
| 1. Capture | le texte affiché + une capture |
| 2. Mod | le texte affiché + `getPlayer.json` et `getSchematics.json` |
| 3. Contrôles | les mouvements qui ont marché ou non |
| 4. OCR | l'image annotée `..._ocr.png` |
