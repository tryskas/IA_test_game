# Une IA qui apprend à battre un boss de Hollow Knight

Le but : une IA qui apprend toute seule, par essais et erreurs (apprentissage par renforcement), à battre un boss dans le Panthéon des dieux. Elle refait le combat en boucle : elle gagne des points quand elle touche le boss, elle en perd quand elle est touchée, et garde ce qui marche.

Trois morceaux :

1. **Le mod HKAIBridge** (`mod/HKAIBridge`), dans le jeu : il envoie à Python, à chaque image, l'état du combat (position et vie du chevalier, position et vie du boss) et peut accélérer le jeu pour s'entraîner plus vite.
2. **Les mains** : l'IA appuie sur les touches du clavier comme toi.
3. **L'entraînement** : à venir, une fois les deux premiers vérifiés.

Les commandes se tapent dans PowerShell, depuis le dossier du dépôt, avec l'environnement Python déjà utilisé (`.venv`).

## Étape 1 : installer l'API de mods (Lumafly)

1. Télécharge et lance **Lumafly**, le gestionnaire de mods de Hollow Knight : https://themulhima.github.io/Lumafly/
2. Il trouve le jeu tout seul (sinon, indique-lui le dossier de Hollow Knight).
3. Installe l'**API de mods** : bouton en bas de la fenêtre de Lumafly (« Installer l'API » ou « Activer l'API »). Installer n'importe quel mod l'installe aussi.
4. Lance le jeu une fois : dans le menu principal, en haut à gauche, la liste des mods doit s'afficher (« Modding API »). Ferme le jeu.

## Étape 2 : installer le SDK .NET

Télécharge et installe le **SDK .NET 8** (gratuit, Microsoft) : https://dotnet.microsoft.com/download

Vérifie dans un nouveau PowerShell :

```bash
dotnet --version
```

## Étape 3 : compiler le mod

```bash
dotnet build hollow_knight/mod/HKAIBridge -c Release
```

La compilation copie le mod dans le dossier des mods du jeu. Si le jeu n'est pas dans le dossier Steam par défaut, donne son dossier `Managed` :

```bash
dotnet build hollow_knight/mod/HKAIBridge -c Release -p:HollowKnightManaged="D:\SteamLibrary\steamapps\common\Hollow Knight\hollow_knight_Data\Managed"
```

Lance le jeu : en haut à gauche du menu principal, « HKAIBridge : 0.1.0 » doit apparaître.

## Étape 4 : vérifier que l'IA reçoit l'état du jeu

Charge ta partie, va au Panthéon des dieux, puis :

```bash
python hollow_knight/check_bridge.py --vitesse
```

Pendant 15 s, le script affiche la scène, la position et la vie du chevalier. Lance un combat contre un boss pendant ce temps : son nom et sa vie s'affichent aussi. Ensuite, le jeu tourne deux fois plus vite pendant 5 s, puis revient à la normale.

## Étape 5 : vérifier les touches

Place le chevalier sur un sol plat avec de la place des deux côtés (le Panthéon des dieux convient), puis :

```bash
python hollow_knight/check_keys.py
```

Clique dans le jeu pendant le compte à rebours. L'IA va à droite, à gauche, saute, fait une ruée et attaque. Le script dit ce qui a marché. Si une touche ne marche pas, regarde dans le jeu (Options > Clavier) quelle touche fait cette action et écris-la dans `hollow_knight/hk_config.py`.

## Ce que tu m'envoies

Le texte affiché par `check_bridge.py` (avec un boss en combat) et par `check_keys.py`, et le nom du boss sur lequel tu veux entraîner l'IA en premier.
