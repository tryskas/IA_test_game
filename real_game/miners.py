"""Portable miners (« Foreuse portable »): placed on a node, they mine by themselves, about
40 ore a minute each on a normal node, far faster than by hand.

Learnt from the demo of 30/10/2026 (demos/demo_20261030_134713):
- equip: inventory (Tab), double-click on the miners among « Raccourcis d'équipement »;
- place: aim at the node until the game says « ... pour placer la foreuse sur le gisement »,
  left click; turn a little and do it again for the next one;
- collect: look at a miner, E opens its window « Foreuse portable », the button above
  « Tout récupérer » puts its ore in the inventory; close the window;
- take back: same, with the button above « Ramasser la foreuse » (its ore comes too, and the
  window closes by itself).
Where each miner was placed from (position, facing, pitch) is remembered in memory.json.
"""
from __future__ import annotations

from config import KEYS
from menus import Memory, Menus, no_sightings
from moves import Mover, wrap

MINER_CLASS = "BP_ItemDescriptorPortableMiner_C"
TITLE = "Foreuse portable"
TAKE_ALL, TAKE_BACK = "Tout récupérer", "Ramasser la foreuse"
PLACE_WORDS = "placer la foreuse"
BUTTON_ABOVE = 75  # reference pixels: each button is this far above its label
PLACE_TURNS = (0, -25, 25, -50, 50, -75, 75)  # degrees around the node, one miner per direction
PLACE_PITCHES = (45, 35, 55, 25, 65)  # degrees down, looking for a free spot on the node
PER_MINUTE = 40.0  # ore a miner gives on a normal node
COLLECT_EVERY_S = 45.0  # seconds between two rounds of collecting
STAND_RADIUS = 1.0  # metres: back where a miner was placed from


class PortableMiners:
    def __init__(self, game, mover: Mover, menus: Menus, memory: Memory, log=print):
        self.game, self.mover, self.menus, self.memory, self.log = game, mover, menus, memory, log

    @property
    def placed(self) -> list[dict]:
        """Miners standing on nodes: {resource, x, y, facing, pitch} of where each was placed from."""
        return self.memory.data.setdefault("miners", [])

    def on(self, resource: str) -> list[dict]:
        return [m for m in self.placed if m["resource"] == resource]

    def carried(self) -> int:
        return int(self.game.player().by_class.get(MINER_CLASS, 0))

    # Placing ---------------------------------------------------------------------------

    def place(self, resource: str) -> int:
        """Put every miner carried on the nearest free node of `resource`. Returns how many."""
        count = self.carried()
        if not count:
            return 0
        node = self.mover.go_to_node(resource)
        self.mover.face(node.x, node.y)
        placed = 0
        with no_sightings(self.game):  # the miner's hologram can look like a creature to the eyes
            self.menus.equip(MINER_CLASS)
            try:
                base = self.mover.facing()
                for turn in PLACE_TURNS:
                    if placed >= count:
                        break
                    self.mover.turn(wrap(base + turn - self.mover.facing()))
                    for pitch in PLACE_PITCHES:
                        self.mover.set_pitch(pitch)
                        self.game.wait(0.3)
                        if self.menus.screen.find(PLACE_WORDS, "center") is None:
                            continue
                        self.game.click()
                        self.game.wait(0.8)
                        player = self.game.player()
                        self.placed.append({"resource": resource, "x": player.x, "y": player.y,
                                            "facing": self.mover.facing(player), "pitch": pitch})
                        placed += 1
                        break
            finally:
                self.memory.save()
                self.menus.equip_xeno_zapper()  # back to the zapper, to be able to defend
        self.log(f"  {placed} foreuse(s) posée(s)")
        return placed

    # Using them ------------------------------------------------------------------------

    def _open(self, miner: dict) -> bool:
        """Go back where a miner was placed from, look at it, open its window."""
        if not self.mover.walk_to(miner["x"], miner["y"], radius=STAND_RADIUS, timeout=60):
            return False
        self.mover.turn(wrap(miner["facing"] - self.mover.facing()))
        self.mover.set_pitch(miner["pitch"])
        self.game.wait(0.3)
        self.menus.press(KEYS["interact"])
        return self.menus.screen.wait_for(TITLE, "machine", timeout=3.0) is not None

    def _button(self, label: str) -> bool:
        found = self.menus.screen.find(label, "machine")
        if found is None:
            return False
        self.menus.click_at(found.x, found.y - BUTTON_ABOVE * self.menus.sy)
        self.game.wait(0.5)
        return True

    def _close_if_open(self) -> None:
        if self.menus.screen.find(TITLE, "machine") is not None:
            self.menus.close()

    def collect(self, resource: str) -> int:
        """Take the ore of every miner on `resource`. Returns how much came."""
        before = self.mover.amount(resource)
        with no_sightings(self.game):
            for miner in self.on(resource):
                if not self._open(miner):
                    self.log("  je ne retrouve pas une foreuse")
                    self._close_if_open()
                    continue
                try:
                    self._button(TAKE_ALL)
                finally:
                    self._close_if_open()
        return int(self.mover.amount(resource) - before)

    def take_back(self, resource: str) -> int:
        """Pick up the miners on `resource`, with their ore. Returns how many came back."""
        before = self.carried()
        kept = [m for m in self.placed if m["resource"] != resource]
        with no_sightings(self.game):
            for miner in self.on(resource):
                had = self.carried()
                if self._open(miner):
                    self._button(TAKE_BACK)  # the window closes by itself
                self._close_if_open()
                if self.carried() <= had:
                    kept.append(miner)  # still there: another time
        self.memory.data["miners"] = kept
        self.memory.save()
        return self.carried() - before

    def mine(self, resource: str, count: int) -> int | None:
        """Get `count` of `resource` with the miners: bring back those working on another ore,
        place them here, collect until there is enough. None if there are no miners."""
        for other in {m["resource"] for m in self.placed} - {resource}:
            self.log(f"  je reprends mes foreuses ({other})")
            self.take_back(other)
        just_placed = self.place(resource) if self.carried() else 0
        working = len(self.on(resource))
        if not working:
            return None
        start = self.mover.amount(resource)
        self.log(f"  {working} foreuse(s) au travail, je récupère le minerai au fur et à mesure")
        deadline = self.game.now() + 60 + 2 * 60 * count / (PER_MINUTE * working)
        if just_placed:
            self.game.wait(COLLECT_EVERY_S)  # nothing to take yet
        while self.game.now() < deadline:
            self.collect(resource)
            if self.mover.amount(resource) - start >= count:
                break
            self.game.wait(COLLECT_EVERY_S)
        return int(self.mover.amount(resource) - start)
