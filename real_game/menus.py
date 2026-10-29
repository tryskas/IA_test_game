"""Using the game's menus: inventory, build menu, workbench.

Buttons are found by reading the text on screen, so they are found even when
lists change as new recipes unlock. Every action is checked through the mod.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from config import KEYS
from moves import Mover, SkillError, wrap
from screen import Screen, normalize

try:
    import inputs
except ImportError:  # not on Windows: the menus can't be used, but their settings can be tested
    inputs = None

MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"
REFERENCE_WIDTH, REFERENCE_HEIGHT = 2560, 1440  # screen size the offsets below were measured on
ZAPPER_OFFSET = (61, 49)  # from the left of « Raccourcis d'équipement » to the first equipment slot
BUILD_TILES_BELOW = 0.21  # build menu: results are under this height (fraction of the screen)
PLACE_PITCH = 20  # degrees below the horizon when placing a building
CRAFT_BUTTON = (1198, 940)  # workbench « Fabriquer » button, on the reference screen


class Memory:
    """Where the AI built things, to come back to them (one file per save)."""

    def __init__(self, data: dict | None = None):
        self.data = data or {"hub_id": None, "spots": {}}

    @classmethod
    def load(cls) -> "Memory":
        if MEMORY_FILE.exists():
            return cls(json.loads(MEMORY_FILE.read_text(encoding="utf-8")))
        return cls()

    def save(self) -> None:
        MEMORY_FILE.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    @property
    def spots(self) -> dict:
        return self.data["spots"]

    def check_save(self, hub_id: str | None) -> None:
        """Forget everything when the HUB is not the one remembered (another save)."""
        if hub_id and self.data.get("hub_id") not in (None, hub_id):
            self.data = {"hub_id": hub_id, "spots": {}}
        elif hub_id:
            self.data["hub_id"] = hub_id
        self.save()


class Menus:
    def __init__(self, raw, mover: Mover, screen: Screen, memory: Memory, log=print):
        self.raw, self.mover, self.screen, self.memory, self.log = raw, mover, screen, memory, log
        self.sx = screen.width / REFERENCE_WIDTH
        self.sy = screen.height / REFERENCE_HEIGHT

    # Small gestures ---------------------------------------------------------------

    def press(self, key: str, seconds: float = 0.06) -> None:
        self.raw.hold(key)
        self.raw.wait(seconds)
        self.raw.release(key)
        self.raw.wait(0.25)

    def click_at(self, x: float, y: float, double: bool = False) -> None:
        self.raw.check()
        inputs.move_to(x, y)
        self.raw.wait(0.1)
        if double:
            inputs.double_click()
        else:
            inputs.click()
        self.raw.wait(0.3)

    def type(self, text: str) -> None:
        self.raw.check()
        inputs.type_text(text)
        self.raw.wait(0.6)

    def amount(self, item: str) -> float:
        return self.raw.player().inventory.get(item, 0)

    # Inventory ------------------------------------------------------------------------

    def equip_xeno_zapper(self) -> bool:
        """Put the Xeno-Zapper in hand. Returns False if it already was."""
        def carried() -> bool:
            return any(normalize(name) == "repoussexenos" for name in self.raw.player().other_items)

        if not carried():
            return False
        self.press(KEYS["inventory"])
        label = self.screen.wait_for("Raccourcis d'équipement", "inventory")
        if label is None:
            self.press(KEYS["close"])
            raise SkillError("l'inventaire ne s'ouvre pas")
        self.click_at(label.left + ZAPPER_OFFSET[0] * self.sx, label.y + ZAPPER_OFFSET[1] * self.sy, double=True)
        self.raw.wait(0.5)
        self.press(KEYS["inventory"])
        if carried():
            raise SkillError("le repousse-xénos n'a pas été équipé")
        return True

    # Build menu -------------------------------------------------------------------------

    def build(self, name: str, done: Callable[[], bool], spot: str | None = None, tries: int = 4) -> None:
        """Open the build menu, pick `name`, place it in front, check with `done()`.

        `spot`: remember where we stood, to come back and use the building later.
        """
        for attempt in range(tries):
            self.press(KEYS["build"])
            if self.screen.wait_for("Constructeur", "build_menu") is None:
                raise SkillError("le menu de construction ne s'ouvre pas")
            self.press("space")  # search field
            self.type(name)
            tile = self.screen.find(name, "build_menu", below=self.screen.top + BUILD_TILES_BELOW * self.screen.height)
            if tile is None:
                self.press(KEYS["close"])
                raise SkillError(f"« {name} » n'est pas dans le menu de construction")
            self.click_at(tile.x, tile.y)
            self.mover.set_pitch(PLACE_PITCH)
            self.raw.wait(0.4)
            self.raw.check()
            inputs.click()  # place the hologram where the camera looks
            placed = self._wait_until(done, seconds=6)  # the mod sees new buildings a few seconds later
            self.press(KEYS["close"])  # leave build mode
            if placed:
                if spot:
                    player = self.raw.player()
                    self.memory.spots[spot] = {"x": player.x, "y": player.y,
                                               "facing": self.mover.facing(player), "pitch": PLACE_PITCH}
                    self.memory.save()
                return
            self.log(f"  impossible de poser ici (essai {attempt + 1}), je tourne")
            self.mover.turn(70)
        raise SkillError(f"je n'arrive pas à poser : {name}")

    def _wait_until(self, done: Callable[[], bool], seconds: float) -> bool:
        end = self.raw.now() + seconds
        while True:
            self.raw.wait(0.5)
            if done():
                return True
            if self.raw.now() >= end:
                return False

    def spend_check(self, cost: dict[str, float]) -> Callable[[], bool]:
        """A `done()` for build(): true once the building's cost left the inventory."""
        before = {item: self.amount(item) for item in cost}

        def spent() -> bool:
            inventory = self.raw.player().inventory
            return all(before[item] - inventory.get(item, 0) >= qty for item, qty in cost.items())

        return spent

    # Workbench -----------------------------------------------------------------------------

    def open_workbench(self) -> None:
        spot = self.memory.spots.get("workbench")
        if spot is None:
            raise SkillError("je ne sais pas où est l'établi")
        if not self.mover.walk_to(spot["x"], spot["y"], radius=1.0):
            raise SkillError("je n'arrive pas à retourner à l'établi")
        self.mover.turn(wrap(spot["facing"] - self.mover.facing()))
        for pitch in (spot["pitch"], spot["pitch"] - 10, spot["pitch"] + 10, spot["pitch"] + 20, 5):
            self.mover.set_pitch(pitch)
            self.press(KEYS["interact"])
            if self.screen.wait_for("Chercher une formule", "workbench", timeout=2.5):
                return
        raise SkillError("je n'arrive pas à ouvrir l'établi")

    def craft(self, recipe: str, product: str, count: int) -> int:
        """Craft `count` × `recipe` (its in-game name) at the workbench. Returns how many were made."""
        start = self.amount(product)
        self.open_workbench()
        try:
            search = self.screen.find("Chercher une formule", "workbench")
            if search is not None:
                self.click_at(search.x, search.y)
                self.type(recipe)
            row = self.screen.find(recipe, "workbench", below=(search.y + 25) if search else None)
            if row is None:
                raise SkillError(f"recette introuvable à l'établi : {recipe}")
            self.click_at(row.x, row.y)
            made = self._hold_to_craft(product, start, count, KEYS["craft"])
            if made == 0:  # the key did nothing: hold the button instead
                self.raw.check()
                inputs.move_to(CRAFT_BUTTON[0] * self.sx + self.screen.left, CRAFT_BUTTON[1] * self.sy + self.screen.top)
                made = self._hold_to_craft(product, start, count, None)
            return made
        finally:
            self.press(KEYS["close"])

    def _hold_to_craft(self, product: str, start: float, count: int, key: str | None) -> int:
        """Hold the craft key (or the mouse button) until enough is made or nothing happens."""
        if key:
            self.raw.hold(key)
        else:
            self.raw.check()
            inputs.mouse_down()
        try:
            last, still = self.amount(product), 0.0
            while last < start + count and still < 5.0:
                self.raw.wait(0.5)
                now = self.amount(product)
                still = 0.0 if now > last else still + 0.5
                last = now
        finally:
            if key:
                self.raw.release(key)
            else:
                inputs.mouse_up()
        return int(self.amount(product) - start)
