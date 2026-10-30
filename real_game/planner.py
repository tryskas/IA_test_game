"""Game plan for the HUB milestones, worked out from the game's own data.

The mod gives every milestone's cost (and what is still missing), every recipe
(ingredients, products, where it can be made) and what is unlocked. From that,
the planner decides the next step: choose a milestone at the HUB terminal,
mine, craft at the workbench, or go and deposit.

Everything is done by hand for now: machines come later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Raw resources mined by hand (game class name -> resource name used by moves.py).
RAW = {"Desc_OreIron_C": "iron_ore", "Desc_OreCopper_C": "copper_ore", "Desc_Stone_C": "limestone"}
WORKBENCH = "BP_WorkBenchComponent_C"
MILESTONE_TYPES = ("Tutorial", "Milestone")
MINE_MAX = 300  # ore mined per trip to a node (three inventory stacks)
CRAFT_BATCH = 500  # crafts per visit to the workbench: everything needed at once


@dataclass(frozen=True)
class Recipe:
    name: str  # in-game name, as written at the workbench
    inputs: dict[str, float]
    outputs: dict[str, float]
    seconds: float  # one craft by hand


@dataclass(frozen=True)
class Milestone:
    cls: str
    name: str
    tier: int
    kind: str
    purchased: bool
    locked: bool
    remaining: dict[str, float]  # still to deposit, by item class
    total: dict[str, float]


@dataclass
class Plan:
    """What the game allows right now."""

    milestones: dict[str, Milestone]
    recipes: dict[str, Recipe]  # handcraftable and unlocked, by the item they make
    names: dict[str, str] = field(default_factory=dict)  # item class -> in-game name

    def name(self, cls: str) -> str:
        return self.names.get(cls, cls)


@dataclass(frozen=True)
class Step:
    kind: str  # select | deposit | mine | craft | wait | done | stuck
    milestone: Milestone | None = None
    recipe: Recipe | None = None
    item: str | None = None  # item class: what is mined or crafted
    count: int = 0  # ore to mine, or crafts to do
    why: str = ""


def parse_schematics(entries: list[dict]) -> Plan:
    milestones, recipes, names = {}, {}, {}
    for s in entries:
        for c in s.get("Cost", []):
            names[c["ClassName"]] = c.get("Name", c["ClassName"])
        if s.get("Type") in MILESTONE_TYPES:
            milestones[s["ClassName"]] = Milestone(
                s["ClassName"], s.get("Name", s["ClassName"]), int(s.get("TechTier") or 0), s["Type"],
                bool(s.get("Purchased")), bool(s.get("Locked")),
                {c["ClassName"]: c.get("RemainingCost", c["Amount"]) for c in s.get("Cost", [])},
                {c["ClassName"]: c.get("TotalCost", c["Amount"]) for c in s.get("Cost", [])},
            )
        if not s.get("Purchased"):
            continue
        for r in s.get("Recipes", []):
            for part in (*r.get("Ingredients", []), *r.get("Products", [])):
                names[part["ClassName"]] = part.get("Name", part["ClassName"])
            if WORKBENCH not in r.get("ProducedIn", []) or len(r.get("Products", [])) != 1:
                continue
            if r["ClassName"].startswith("Recipe_Alternate"):
                continue  # keep the standard recipes
            product = r["Products"][0]["ClassName"]
            recipes.setdefault(product, Recipe(
                r["Name"],
                {i["ClassName"]: i["Amount"] for i in r["Ingredients"]},
                {p["ClassName"]: p["Amount"] for p in r["Products"]},
                float(r.get("ManualDuration") or 1),
            ))
    return Plan(milestones, recipes, names)


def raw_cost(plan: Plan, item: str, amount: float, depth: int = 0) -> float:
    """Ore that goes into `amount` of `item` (infinite if it can't be made by hand)."""
    if item in RAW:
        return amount
    recipe = plan.recipes.get(item)
    if recipe is None or depth > 10:
        return math.inf
    crafts = amount / recipe.outputs[item]
    return sum(raw_cost(plan, i, q * crafts, depth + 1) for i, q in recipe.inputs.items())


BASE_PAIR_MAX = 150.0  # metres between the iron and the copper node of a base, at most
BASE_SEARCH = 800.0  # metres from the player where a base is looked for
PURITY_BONUS = {"pure": -15.0, "normal": 0.0, "impure": 15.0}  # metres a node's purity is worth


def choose_base(nodes, x: float, y: float) -> tuple[float, float, float, float] | None:
    """Where to build the HUB: between an iron and a copper node close to each other, at about the
    same height, near limestone and not too far from (x, y). Returns (x, y, metres between the two
    nodes, metres from here), or None if no such pair."""
    free = [n for n in nodes if not n.exploited]
    stones = [n for n in free if n.resource == "limestone"]
    best = None
    for iron in (n for n in free if n.resource == "iron_ore"):
        for copper in (n for n in free if n.resource == "copper_ore"):
            apart = math.hypot(iron.x - copper.x, iron.y - copper.y)
            mx, my = (iron.x + copper.x) / 2, (iron.y + copper.y) / 2
            trip = math.hypot(mx - x, my - y)
            if apart > BASE_PAIR_MAX or trip > BASE_SEARCH:
                continue
            stone = min((math.hypot(s.x - mx, s.y - my) for s in stones), default=BASE_PAIR_MAX)
            score = (apart + 2 * abs(iron.z - copper.z) + 0.5 * min(stone, BASE_PAIR_MAX) + 0.2 * trip
                     + PURITY_BONUS.get(iron.purity, 0) + PURITY_BONUS.get(copper.purity, 0))
            if best is None or score < best[0]:
                best = (score, mx, my, apart, trip)
    return None if best is None else best[1:]


def raw_needs(plan: Plan, wanted: dict[str, float], inventory: dict[str, float]) -> dict[str, float] | None:
    """Ore still to mine to make everything `wanted`, using first what is carried (the items
    themselves, then their parts, down to the ore). None if something can't be made by hand."""
    stock = dict(inventory)
    needs: dict[str, float] = {}

    def need(item: str, amount: float, depth: int) -> bool:
        use = min(stock.get(item, 0), amount)
        stock[item] = stock.get(item, 0) - use
        amount -= use
        if amount <= 1e-9:
            return True
        if item in RAW:
            needs[item] = needs.get(item, 0) + amount
            return True
        recipe = plan.recipes.get(item)
        if recipe is None or depth > 10:
            return False
        crafts = math.ceil(amount / recipe.outputs[item] - 1e-9)
        return all(need(part, per_craft * crafts, depth + 1) for part, per_craft in recipe.inputs.items())

    for item, amount in wanted.items():
        if not need(item, amount, 0):
            return None
    return needs


def choose_milestone(plan: Plan) -> Milestone | None:
    """The next milestone to work on: lowest tier first, then the cheapest in ore."""
    candidates = [
        m for m in plan.milestones.values()
        if not m.purchased and not m.locked
        and all(raw_cost(plan, item, qty) < math.inf for item, qty in m.total.items())
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda m: (m.tier, sum(raw_cost(plan, i, q) for i, q in m.remaining.items())))


def obtain(plan: Plan, item: str, amount: float, inventory: dict[str, float], depth: int = 0) -> Step:
    """Next step to get `amount` more of `item`."""
    if item in RAW:
        return Step("mine", item=item, count=min(max(math.ceil(amount), 10), MINE_MAX))
    recipe = plan.recipes.get(item)
    if recipe is None or depth > 10:
        return Step("stuck", item=item, why=f"je ne sais pas fabriquer : {plan.name(item)}")
    crafts = min(math.ceil(amount / recipe.outputs[item]), CRAFT_BATCH)
    for ingredient, per_craft in recipe.inputs.items():
        have = inventory.get(ingredient, 0)
        if have < per_craft * crafts:
            return obtain(plan, ingredient, per_craft * crafts - have, inventory, depth + 1)
    return Step("craft", recipe=recipe, item=item, count=crafts)


def next_step(plan: Plan, inventory: dict[str, float], active: str | None, ship_docked: bool = True) -> Step:
    """The next thing to do, given what we carry and the HUB terminal's state."""
    if active is None:
        if not ship_docked:
            return Step("wait", why="la navette du HUB n'est pas revenue")
        target = choose_milestone(plan)
        if target is None:
            return Step("done", why="plus aucun jalon faisable à la main")
        return Step("select", milestone=target)
    target = plan.milestones[active]
    remaining = {item: qty for item, qty in target.remaining.items() if qty > 0}
    if not remaining:
        return Step("wait", milestone=target, why="jalon complet, en attente de validation")
    if all(inventory.get(item, 0) >= qty for item, qty in remaining.items()):
        return Step("deposit", milestone=target)
    # Mine all the ore the milestone still needs first (one trip per ore), then craft it all.
    needs = {ore: qty for ore, qty in (raw_needs(plan, remaining, inventory) or {}).items() if qty > 1e-6}
    if needs:
        ore = max(needs, key=needs.get)
        return Step("mine", target, item=ore, count=min(max(math.ceil(needs[ore]), 10), MINE_MAX),
                    why=f"il faut encore {math.ceil(needs[ore])} {plan.name(ore)} en tout")
    # Make the dearest missing item first: it takes longest, the rest comes on the way.
    missing = {item: qty - inventory.get(item, 0) for item, qty in remaining.items() if inventory.get(item, 0) < qty}
    item = max(missing, key=lambda i: raw_cost(plan, i, missing[i]))
    step = obtain(plan, item, missing[item], inventory)
    return Step(step.kind, target, step.recipe, step.item, step.count, step.why)
