"""Step 2, exact game data: read the game through the Ficsit Remote Monitoring mod.

The mod runs a small web server inside the game that gives the game's data
(player, inventory, power, machines...) as JSON. This script asks for each kind
of data, says which ones answer, and saves them in captures/frm.

Before running it: game launched with the mod, save loaded, and in the game
chat (Enter key): /frm http start

    python real_game/check_game_data.py
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from config import CAPTURE_DIR, FRM_URL

# Data asked to the mod. The exact list depends on the mod's version: the ones
# it does not know simply come back as "unknown".
ENDPOINTS = {
    "getSessionInfo": "la partie en cours",
    "getPlayer": "le joueur : position, inventaire",
    "getPower": "l'électricité, circuit par circuit",
    "getFactory": "toutes les machines",
    "getProdStats": "la production par objet",
    "getWorldInv": "tout ce qui est stocké",
    "getStorageInv": "le contenu des conteneurs",
    "getSchematics": "les jalons et ce qui est débloqué",
    "getHUBTerminal": "le terminal du HUB",
    "getResourceNode": "les gisements",
}


def ask(endpoint: str):
    with urllib.request.urlopen(f"{FRM_URL}/{endpoint}", timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def describe(value) -> str:
    if isinstance(value, list):
        keys = sorted(value[0].keys())[:8] if value and isinstance(value[0], dict) else []
        return f"{len(value)} éléments" + (f", champs : {', '.join(keys)}" if keys else "")
    if isinstance(value, dict):
        return f"champs : {', '.join(sorted(value.keys())[:8])}"
    return str(value)[:80]


def main() -> None:
    folder = CAPTURE_DIR / "frm"
    folder.mkdir(parents=True, exist_ok=True)
    answered = 0
    for endpoint, meaning in ENDPOINTS.items():
        try:
            value = ask(endpoint)
        except urllib.error.HTTPError as error:
            print(f"  {endpoint:<16} inconnu de cette version du mod (erreur {error.code})")
            continue
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            raise SystemExit(
                f"Le mod ne répond pas à {FRM_URL}. Vérifie que le jeu a été lancé avec le mod "
                "Ficsit Remote Monitoring, qu'une partie est chargée, et tape dans le chat du jeu : "
                "/frm http start"
            )
        except json.JSONDecodeError:
            print(f"  {endpoint:<16} réponse illisible")
            continue
        answered += 1
        path = folder / f"{endpoint}.json"
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"OK {endpoint:<16} {meaning} : {describe(value)}")
    print(f"\n{answered} types de données lus, enregistrés dans {folder}")


if __name__ == "__main__":
    main()
