# IA_test_game : une IA qui apprend à battre un boss de Hollow Knight

Une IA qui apprend toute seule, par essais et erreurs (apprentissage par renforcement), à battre un boss du Panthéon des dieux de Hollow Knight. Elle refait le combat en boucle : elle gagne des points quand elle touche le boss, elle en perd quand elle est touchée, et garde ce qui marche.

Tout est dans le dossier [`hollow_knight/`](hollow_knight/). L'installation pas à pas est dans [son guide](hollow_knight/GUIDE.md).

## Contenu

| Fichier | Rôle |
|---|---|
| `hollow_knight/mod/HKAIBridge/` | Le mod du jeu (C#) : il envoie l'état du combat à Python et peut accélérer le jeu |
| `hollow_knight/hk_bridge.py` | Réception de l'état du jeu côté Python |
| `hollow_knight/hk_inputs.py` | Les touches du clavier, envoyées au jeu |
| `hollow_knight/hk_config.py` | Les réglages : touches du jeu, ports |
| `hollow_knight/check_bridge.py`, `check_keys.py` | Vérifier le mod et les touches |
| `tests/` | Tests automatiques |

## Tests

```bash
python -m unittest discover -s tests -t .
```
