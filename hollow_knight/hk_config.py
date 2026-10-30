"""Réglages à adapter à ton jeu.

Touches : celles de Hollow Knight par défaut (Options > Clavier). Si tu les as changées
dans le jeu, change-les ici aussi. On écrit la lettre affichée dans le jeu ; les flèches
s'écrivent "left", "right", "up", "down".
"""
KEYS = {
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "jump": "z",  # Sauter
    "attack": "x",  # Attaquer
    "dash": "c",  # Ruée
    "cast": "a",  # Concentration / sort
}

STOP_KEY = "f12"  # arrête tout, à tout moment
STATE_PORT = 9999  # le mod envoie l'état du jeu ici
COMMAND_PORT = 9998  # le mod écoute les commandes ici (vitesse du jeu)
