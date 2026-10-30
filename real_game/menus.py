"""Using the game's menus: inventory, build menu, workbench.

Buttons are found by reading the text on screen, so they are found even when
lists change as new recipes unlock. Every action is checked through the mod.
"""
from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from config import KEYS
from moves import EYE_HEIGHT, Mover, SkillError, wrap
from screen import Screen, normalize

try:
    import inputs
except ImportError:  # not on Windows: the menus can't be used, but their settings can be tested
    inputs = None

MEMORY_FILE = Path(__file__).resolve().parent / "memory.json"
HEIGHTS_FILE = Path(__file__).resolve().parent / "heights.npz"  # the 3D map of the ground
REFERENCE_WIDTH, REFERENCE_HEIGHT = 2560, 1440  # screen size the offsets below were measured on
ZAPPER_OFFSET = (61, 49)  # from the left of « Raccourcis d'équipement » to the first equipment slot
BUILD_TILES_BELOW = 0.21  # build menu: results are under this height (fraction of the screen)
PLACE_TURNS = (0, 45, -45, 90, -90, 135, -135, 180)  # directions tried to find room for a building
PLACE_PITCHES = (25, 20, 30, 15)  # degrees below the horizon: about 3 to 6 m ahead, near first
USE_DISTANCES = (3.0, 2.2, 4.0)  # metres from a building's centre to stand when using it
USE_PITCHES = (10, 20, 0, 30)  # degrees below the horizon to look at a building
PLACE_MOVES = 3  # walks forward to look for room elsewhere
CRAFT_BUTTON = (1198, 940)  # workbench « Fabriquer » button, on the reference screen
FIRST_RECIPE = (600, 397)  # workbench: first recipe under the search field, on the reference screen
# HUB terminal, from the « Choisir une étape » tab: tier tabs, and the « Étape active » tab.
TIER_TAB = (115, 90, 132)  # first tab offset x, y, then gap between tabs
ACTIVE_TAB = (260, 0)
MILESTONES_BELOW = 200  # milestone names are this far under the tabs
# From the left of « Objets pertinents »: first item the player carries for the milestone.
RELEVANT_SLOT = (31, 68, 104)  # offset x, y, then gap between items
SUBMIT_WORDS = ("Envoyer", "Expédier", "Soumettre", "Valider", "Livrer")
TERMINAL_DISTANCE = 2.5
TERMINAL_SIDES = (0, 45, -45, 90, -90, 180)  # around the terminal, from its front


@contextmanager
def no_alarms(raw):
    """No alarm at all meanwhile (closing a menu): the danger is dealt with right after."""
    before = getattr(raw, "alarms", None)
    if before is not None:
        raw.alarms = False
    try:
        yield
    finally:
        if before is not None:
            raw.alarms = before


@contextmanager
def no_sightings(raw):
    """Meanwhile, creatures seen on screen don't stop the AI (a menu or a hologram is in the
    way of the eyes); losing health still does."""
    before = getattr(raw, "ignore_sightings_until", None)
    if before is not None:
        raw.ignore_sightings_until = float("inf")
    try:
        yield
    finally:
        if before is not None:
            raw.ignore_sightings_until = before


def building_position(spot: dict) -> tuple[float, float]:
    """Where a remembered building stands: given directly, or worked out from where
    the camera looked when it was placed (older memories)."""
    if "building" in spot:
        return tuple(spot["building"])
    distance = EYE_HEIGHT / math.tan(math.radians(spot["pitch"]))
    facing = math.radians(spot["facing"])
    return spot["x"] + distance * math.cos(facing), spot["y"] + distance * math.sin(facing)


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
            HEIGHTS_FILE.unlink(missing_ok=True)
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
        inputs.glide_to(x, y)
        self.raw.wait(0.08)
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
        """How many of an item (game class name, like Desc_IronRod_C) the player carries."""
        return self.raw.player().by_class.get(item, 0)

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

    def build(self, name: str, done: Callable[[], bool], spot: str | None = None) -> None:
        """Open the build menu, pick `name`, place it where there is room, check with `done()`.

        `spot`: remember where we stood, to come back and use the building later.
        """
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
        try:
            with no_sightings(self.raw):  # the hologram can look like a creature to the eyes
                pitch = self._place(done)
        finally:
            self.close()  # leave build mode
        if spot:
            player = self.raw.player()
            placed_at = {"x": player.x, "y": player.y, "facing": self.mover.facing(player), "pitch": pitch}
            self.memory.spots[spot] = {"building": list(building_position(placed_at))}
            self.memory.save()

    def _place(self, done: Callable[[], bool]) -> float:
        """Look around for a spot where the hologram is blue, click there. Returns the pitch used."""
        seen = False
        for move in range(PLACE_MOVES):
            base = self.mover.facing()
            for turn in PLACE_TURNS:
                self.mover.turn(wrap(base + turn - self.mover.facing()))
                for pitch in PLACE_PITCHES:
                    self.mover.set_pitch(pitch)
                    self.raw.wait(0.3)
                    state = self.screen.hologram()
                    seen = seen or state != "none"
                    if state != "valid":
                        continue
                    self.raw.check()
                    inputs.click()
                    if self._wait_until(done, seconds=6):  # the mod sees new buildings a few seconds later
                        return pitch
                    self.log("  clic sans effet, je cherche ailleurs")
            if not seen:
                raise SkillError("aucun hologramme à l'écran : le bâtiment n'a pas été sélectionné")
            self.log("  pas de place libre ici, j'avance un peu")
            self.mover.set_pitch(0)
            self.raw.hold(KEYS["forward"])
            self.raw.wait(1.5)
            self.raw.release(KEYS["forward"])
        raise SkillError("aucun emplacement libre trouvé")

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
            inventory = self.raw.player().by_class
            return all(before[item] - inventory.get(item, 0) >= qty for item, qty in cost.items())

        return spent

    def open_workbench(self) -> None:
        spot = self.memory.spots.get("workbench")
        if spot is None:
            raise SkillError("je ne sais pas où est l'établi")
        x, y = building_position(spot)
        for distance in USE_DISTANCES:  # come closer if it does not open
            if not self.mover.walk_to(x, y, radius=distance):
                raise SkillError("je n'arrive pas à aller jusqu'à l'établi")
            self.mover.face(x, y)
            for pitch in USE_PITCHES:
                self.mover.set_pitch(pitch)
                self.press(KEYS["interact"])
                if self.screen.wait_for("Chercher une formule", "workbench", timeout=2.0):
                    return
        raise SkillError("je n'arrive pas à ouvrir l'établi")

    # HUB terminal ---------------------------------------------------------------------------

    def open_terminal(self, terminal: dict) -> None:
        """Walk to the HUB terminal (as the mod gives it) and open it."""
        tx, ty = terminal["location"]["x"] / 100, terminal["location"]["y"] / 100
        front = self.mover.cal.facing(float(terminal["location"].get("rotation", 0)))
        known = self.memory.spots.get("terminal")
        stands = [tuple(known["stand"])] if known else []
        stands += [(tx + TERMINAL_DISTANCE * math.cos(math.radians(front + side)),
                    ty + TERMINAL_DISTANCE * math.sin(math.radians(front + side))) for side in TERMINAL_SIDES]
        for x, y in stands:
            if not self.mover.walk_to(x, y, radius=0.8):
                continue
            self.mover.face(tx, ty)
            for pitch in USE_PITCHES:
                self.mover.set_pitch(pitch)
                self.press(KEYS["interact"])
                if self.screen.wait_for("Choisir une étape", "terminal_top", timeout=2.0):
                    self.memory.spots["terminal"] = {"stand": [x, y]}
                    self.memory.save()
                    return
        raise SkillError("je n'arrive pas à ouvrir le terminal du HUB")

    def _terminal_anchor(self):
        anchor = self.screen.wait_for("Choisir une étape", "terminal_top", timeout=3.0)
        if anchor is None:
            raise SkillError("le terminal du HUB n'est pas ouvert")
        return anchor

    def select_milestone(self, name: str, tier: int, is_active: Callable[[], bool]) -> None:
        """In the open terminal: pick a tier, a milestone, then confirm. Checked with `is_active()`."""
        anchor = self._terminal_anchor()
        self.click_at(anchor.x, anchor.y)
        dx, dy, gap = TIER_TAB
        self.click_at(anchor.x + (dx + gap * tier) * self.sx, anchor.y + dy * self.sy)
        tile = self.screen.find(name, "terminal_top", below=anchor.y + MILESTONES_BELOW * self.sy)
        if tile is None:
            raise SkillError(f"jalon introuvable dans le terminal : {name}")
        self.click_at(tile.x, tile.y)
        button = self.screen.find("Choisir une étape", "terminal_bottom")
        if button is None:
            raise SkillError("bouton « Choisir une étape » introuvable")
        self.click_at(button.x, button.y)
        if not self._wait_until(is_active, seconds=4):
            raise SkillError(f"le jalon n'a pas été choisi : {name}")

    def deposit(self, is_done: Callable[[], bool]) -> int:
        """In the open terminal: give everything useful the player carries. Returns the stacks given."""
        anchor = self._terminal_anchor()
        self.click_at(anchor.x + ACTIVE_TAB[0] * self.sx, anchor.y + ACTIVE_TAB[1] * self.sy)
        label = self.screen.wait_for("Objets pertinents", "terminal_top", timeout=3.0)
        if label is None:
            raise SkillError("je ne vois pas les objets à déposer")
        dx, dy, gap = RELEVANT_SLOT
        given = 0
        for _ in range(12):  # each deposit reshuffles the row: start again from its first item
            for slot in range(4):
                before = dict(self.raw.player().by_class)
                self.click_at(label.left + (dx + gap * slot) * self.sx, label.y + dy * self.sy, double=True)
                self.raw.wait(0.4)
                after = self.raw.player().by_class
                if any(after.get(item, 0) < qty for item, qty in before.items()):
                    given += 1
                    break
            else:
                break
        if given and not self._wait_until(is_done, seconds=2):
            for word in SUBMIT_WORDS:  # some milestones may need a button to be sent
                button = self.screen.find(word, "terminal_bottom")
                if button is not None:
                    self.click_at(button.x, button.y)
                    break
        return given

    def close(self) -> None:
        with no_alarms(self.raw):  # an alarm must not stop the menu from closing
            self.press(KEYS["close"])

    # Workbench -----------------------------------------------------------------------------

    def craft(self, recipe: str, product: str, count: int) -> int:
        """Make `count` of `product` (class name) with `recipe` (its in-game name) at the workbench.

        Returns how many were made."""
        with no_sightings(self.raw):  # icons in the menu can look like creatures to the eyes
            return self._craft(recipe, product, count)

    def _craft(self, recipe: str, product: str, count: int) -> int:
        start = self.amount(product)
        self.open_workbench()
        try:
            search = self.screen.find("Chercher une formule", "workbench")
            if search is not None:
                self.click_at(search.x, search.y)
                self.type(recipe)
            row = self.screen.find(recipe, "workbench", below=(search.y + 25) if search else None)
            if row is not None:
                self.click_at(row.x, row.y)
            else:  # the search leaves the recipe first in the list, even if its name was misread
                self.click_at(self.screen.left + FIRST_RECIPE[0] * self.sx, self.screen.top + FIRST_RECIPE[1] * self.sy)
            if self.screen.find(recipe, "workbench_recipe") is None:
                self.log(f"  je ne lis pas « {recipe} » comme recette choisie : je vérifierai ce qui est fabriqué")
            made = self._hold_to_craft(product, start, count, KEYS["craft"])
            if made == 0:  # the key did nothing: hold the button instead
                self.raw.check()
                inputs.move_to(CRAFT_BUTTON[0] * self.sx + self.screen.left, CRAFT_BUTTON[1] * self.sy + self.screen.top)
                made = self._hold_to_craft(product, start, count, None)
            return made
        finally:
            self.close()

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
