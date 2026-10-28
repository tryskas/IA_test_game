"""Pygame window that plays recorded runs on screen, side by side as a race."""
from __future__ import annotations

import math
from collections import OrderedDict

import pygame

from mini_satisfactory import data
from mini_satisfactory.replay import NO_INPUT, NO_POWER, OUTPUT_FULL, RUNNING, Frame, MachineView, Run

# Everything is drawn on a fixed-size canvas, then scaled to the window.
W, H = 1280, 800
FPS = 30
SPEEDS = (1, 5, 15, 30, 60, 120, 300)  # game seconds per real second
SEEK_SECONDS = 30

BG = (20, 25, 31)
PANEL = (28, 35, 43)
PANEL_2 = (38, 47, 57)
LINE = (50, 61, 73)
TEXT = (227, 232, 237)
MUTED = (140, 153, 166)
DIM = (92, 104, 116)
ACCENT = (242, 169, 59)  # safety amber
TEAL = (79, 179, 200)  # power and belts
OK = (92, 194, 122)
WARN = (224, 168, 74)
BAD = (239, 107, 91)

ITEM_COLORS = {
    "iron_ore": (124, 139, 156),
    "copper_ore": (196, 112, 64),
    "limestone": (214, 200, 160),
    "leaves": (111, 191, 90),
    "iron_ingot": (160, 178, 197),
    "copper_ingot": (232, 142, 86),
    "iron_plate": (186, 200, 214),
    "iron_rod": (140, 162, 184),
    "screw": (174, 184, 194),
    "wire": (240, 166, 96),
    "cable": (214, 120, 72),
    "concrete": (170, 170, 166),
    "reinforced_iron_plate": (112, 142, 178),
    "portable_miner": ACCENT,
}
STATUS = {
    RUNNING: ("en marche", OK),
    NO_INPUT: ("attend des entrées", WARN),
    OUTPUT_FULL: ("sortie pleine", WARN),
    NO_POWER: ("pas de courant", BAD),
}
RESOURCE_LABELS = {"iron_ore": "Fer", "copper_ore": "Cuivre", "limestone": "Calcaire"}
PURITY_LABELS = {"impure": "impur", "normal": "normal", "pure": "pur"}
UNLOCK_LABELS = {
    "portable_miner": "foreuse portative",
    "smelter": "fonderie",
    "copper_ingot": "cuivre",
    "wire": "fil",
    "cable": "câble",
    "hub_burner": "brûleur du HUB",
    "constructor": "façonneuse",
    "screw": "vis",
    "concrete": "béton",
    "reinforced_iron_plate": "plaque renforcée",
    "conveyor_belt": "convoyeurs",
    "miner_mk1": "foreuse Mk.1",
    "space_elevator": "ascenseur spatial",
    "biomass_burner": "brûleur à biomasse",
}
HELP = "Espace pause · ←/→ ±30 s · ↑/↓ vitesse · 1-4 joueur · Début recommencer · Échap quitter"


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def short_clock(seconds: float) -> str:
    minutes, sec = divmod(int(round(abs(seconds))), 60)
    return f"{minutes:02d}:{sec:02d}"


class Viewer:
    def __init__(self, runs: list[Run], start: int = 0, speed: int = 60, reference: str | None = "usine"):
        self.runs = runs
        self.current = start
        self.reference = next((r for r in runs if r.name == reference), None)
        self.speed_index = min(range(len(SPEEDS)), key=lambda i: abs(SPEEDS[i] - speed))
        self.time = 0.0
        self.paused = False
        self.anim = 0.0  # real seconds of animation, frozen while paused
        self.end_time = max(r.end_time for r in runs)
        self.canvas = pygame.Surface((W, H))
        self.fonts = {
            "title": self._font("bahnschrift,dejavusanscondensed,liberationsansnarrow,arialnarrow,dejavusans", 26, bold=True),
            "clock": self._font("consolas,jetbrainsmono,dejavusansmono,menlo,couriernew", 34, bold=True),
            "h": self._font("bahnschrift,dejavusanscondensed,liberationsansnarrow,arialnarrow,dejavusans", 15, bold=True),
            "big": self._font("segoeui,dejavusans,liberationsans,arial", 19, bold=True),
            "body": self._font("segoeui,dejavusans,liberationsans,arial", 14),
            "bold": self._font("segoeui,dejavusans,liberationsans,arial", 14, bold=True),
            "small": self._font("segoeui,dejavusans,liberationsans,arial", 12),
            "mono": self._font("consolas,jetbrainsmono,dejavusansmono,menlo,couriernew", 13),
        }
        self._text_cache: OrderedDict = OrderedDict()
        self._timeline_rect = pygame.Rect(0, 0, 0, 0)

    @staticmethod
    def _font(names: str, size: int, bold: bool = False) -> pygame.font.Font:
        return pygame.font.SysFont(names, size, bold=bold)

    # ------------------------------------------------------------------ main loop

    def run(self) -> None:
        sizes = pygame.display.get_desktop_sizes()
        screen_w, screen_h = sizes[0] if sizes else (W, H)
        scale = min(1.0, screen_w * 0.92 / W, screen_h * 0.88 / H)
        window = pygame.display.set_mode((int(W * scale), int(H * scale)), pygame.RESIZABLE)
        pygame.display.set_caption("Mini-Satisfactory : l'IA joue")
        ticker = pygame.time.Clock()
        while True:
            real_dt = ticker.tick(FPS) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                self.handle(event, window)
            if not self.paused:
                self.anim += real_dt
                self.time = min(self.end_time, self.time + real_dt * SPEEDS[self.speed_index])
                if self.time >= self.end_time:
                    self.paused = True
            self.draw()
            window = pygame.display.get_surface()
            if window.get_size() == (W, H):
                window.blit(self.canvas, (0, 0))
            else:
                window.fill(BG)
                scale = min(window.get_width() / W, window.get_height() / H)
                scaled = pygame.transform.smoothscale(self.canvas, (int(W * scale), int(H * scale)))
                window.blit(scaled, scaled.get_rect(center=window.get_rect().center))
            pygame.display.flip()

    def handle(self, event: pygame.event.Event, window: pygame.Surface) -> None:
        if event.type == pygame.KEYDOWN:
            key = event.key
            if key == pygame.K_SPACE:
                if self.time >= self.end_time:
                    self.time = 0.0
                self.paused = not self.paused
            elif key == pygame.K_RIGHT:
                self.time = min(self.end_time, self.time + SEEK_SECONDS)
            elif key == pygame.K_LEFT:
                self.time = max(0.0, self.time - SEEK_SECONDS)
            elif key in (pygame.K_UP, pygame.K_PLUS, pygame.K_KP_PLUS):
                self.speed_index = min(len(SPEEDS) - 1, self.speed_index + 1)
            elif key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed_index = max(0, self.speed_index - 1)
            elif key == pygame.K_HOME:
                self.time = 0.0
            elif pygame.K_1 <= key <= pygame.K_9 and key - pygame.K_1 < len(self.runs):
                self.current = key - pygame.K_1
            elif pygame.K_KP1 <= key <= pygame.K_KP9 and key - pygame.K_KP1 < len(self.runs):
                self.current = key - pygame.K_KP1
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            x, y = self._to_canvas(event.pos, window)
            if self._timeline_rect.inflate(0, 14).collidepoint(x, y):
                share = (x - self._timeline_rect.x) / self._timeline_rect.width
                self.time = max(0.0, min(1.0, share)) * self.end_time
            for index, rect in enumerate(getattr(self, "_race_rows", [])):
                if rect.collidepoint(x, y):
                    self.current = index

    @staticmethod
    def _to_canvas(pos: tuple[int, int], window: pygame.Surface) -> tuple[float, float]:
        window = pygame.display.get_surface() or window
        scale = min(window.get_width() / W, window.get_height() / H)
        off_x = (window.get_width() - W * scale) / 2
        off_y = (window.get_height() - H * scale) / 2
        return (pos[0] - off_x) / scale, (pos[1] - off_y) / scale

    # ------------------------------------------------------------------ drawing helpers

    def text(self, s: str, pos, font: str = "body", color=TEXT, anchor: str = "topleft", width: int | None = None):
        key = (s, font, color, width)
        image = self._text_cache.get(key)
        if image is None:
            f = self.fonts[font]
            if width is not None and f.size(s)[0] > width:
                while s and f.size(s + "…")[0] > width:
                    s = s[:-1]
                s += "…"
            image = f.render(s, True, color)
            self._text_cache[key] = image
            if len(self._text_cache) > 3000:
                self._text_cache.popitem(last=False)
        rect = image.get_rect(**{anchor: pos})
        self.canvas.blit(image, rect)
        return rect

    def panel(self, rect: pygame.Rect, title: str) -> pygame.Rect:
        pygame.draw.rect(self.canvas, PANEL, rect, border_radius=6)
        self.text(title.upper(), (rect.x + 14, rect.y + 10), "h", MUTED)
        return rect.inflate(-28, -44).move(0, 16)

    def bar(self, rect: pygame.Rect, share: float, color, striped: bool = False) -> None:
        pygame.draw.rect(self.canvas, PANEL_2, rect, border_radius=3)
        share = max(0.0, min(1.0, share))
        if share <= 0:
            return
        filled = pygame.Rect(rect.x, rect.y, max(3, round(rect.width * share)), rect.height)
        pygame.draw.rect(self.canvas, color, filled, border_radius=3)
        if striped:  # moving stripes: the machine is working
            clip = self.canvas.get_clip()
            self.canvas.set_clip(filled)
            shade = tuple(max(0, c - 45) for c in color)
            offset = int(self.anim * 24) % 12
            for x in range(filled.x - filled.height - 12 + offset, filled.right, 12):
                pygame.draw.line(self.canvas, shade, (x, filled.bottom), (x + filled.height, filled.y), 3)
            self.canvas.set_clip(clip)

    def dot(self, center, color, radius: int = 5) -> None:
        pygame.draw.circle(self.canvas, color, center, radius)

    # ------------------------------------------------------------------ screen

    def draw(self) -> None:
        run = self.runs[self.current]
        frame = run.frame_at(self.time)
        self.canvas.fill(BG)
        self.draw_top_bar(run)
        self.draw_hub(pygame.Rect(12, 76, 300, 550), run, frame)
        self.draw_factory(pygame.Rect(324, 76, 612, 550), frame)
        self.draw_pioneer(pygame.Rect(948, 76, 320, 550), run, frame)
        self.draw_race(pygame.Rect(12, 636, 1256, 156), run)

    def draw_top_bar(self, run: Run) -> None:
        title = self.text("MINI-SATISFACTORY", (16, 14), "title", ACCENT)
        self.text("Tier 0 : les 6 HUB upgrades", (16, 44), "small", MUTED)
        chip = self.text(f"On suit : {run.name}", (title.right + 34, 30), "big", TEXT, anchor="midleft")
        pygame.draw.rect(self.canvas, LINE, chip.inflate(22, 12), 1, border_radius=16)
        self.text(clock(self.time), (W - 190, 30), "clock", TEXT, anchor="midright")
        state = "PAUSE" if self.paused else f"×{SPEEDS[self.speed_index]}"
        self.text(state, (W - 170, 30), "big", ACCENT if self.paused else TEAL, anchor="midleft")
        self.text("vitesse", (W - 170, 46), "small", MUTED)

    def draw_hub(self, rect: pygame.Rect, run: Run, frame: Frame) -> None:
        inner = self.panel(rect, "HUB")
        y = inner.y + 4
        for index, upgrade in enumerate(data.HUB_UPGRADES):
            done = index < frame.upgrades_done
            current = index == frame.upgrades_done
            name_color = TEXT if (done or current) else DIM
            self.text(upgrade.name, (inner.x, y), "bold", name_color)
            if done:
                split = run.upgrade_times[index]
                self.text(short_clock(split), (inner.right, y), "mono", OK, anchor="topright")
                if self.reference is not None and self.reference is not run and index < len(self.reference.upgrade_times):
                    delta = split - self.reference.upgrade_times[index]
                    if abs(delta) < 0.5:
                        gap, color = f"= {self.reference.name}", MUTED
                    else:
                        sign = "-" if delta < 0 else "+"
                        gap, color = f"{sign}{short_clock(delta)} vs {self.reference.name}", OK if delta < 0 else BAD
                    self.text(gap, (inner.right - 52, y + 1), "small", color, anchor="topright")
                unlocks = ", ".join(UNLOCK_LABELS.get(u, u) for u in upgrade.unlocks)
                self.text(f"débloque : {unlocks}", (inner.x, y + 19), "small", MUTED, width=inner.width)
                y += 44
            elif current:
                self.text("en cours", (inner.right, y), "small", ACCENT, anchor="topright")
                y += 22
                for item, qty in upgrade.cost.items():
                    have = min(frame.stock[item], qty)
                    self.text(data.label(item), (inner.x, y), "small", TEXT, width=130)
                    self.bar(pygame.Rect(inner.x + 134, y + 3, inner.width - 210, 10), have / qty, ACCENT)
                    self.text(f"{have:.0f}/{qty:.0f}", (inner.right, y), "mono", MUTED, anchor="topright")
                    y += 19
                y += 10
            else:
                cost = " · ".join(f"{q:.0f} {data.label(i)}" for i, q in upgrade.cost.items())
                self.text(cost, (inner.x, y + 19), "small", DIM, width=inner.width)
                y += 44
            if index < len(data.HUB_UPGRADES) - 1:
                pygame.draw.line(self.canvas, LINE, (inner.x, y - 6), (inner.right, y - 6))
        if frame.upgrades_done >= len(data.HUB_UPGRADES):
            self.text(f"Tier 0 terminé en {short_clock(run.upgrade_times[-1])}", (inner.x, y + 6), "big", OK)

    def draw_factory(self, rect: pygame.Rect, frame: Frame) -> None:
        inner = self.panel(rect, "Usine")
        miners = {(m.target, m.node): m for m in frame.machines if m.node is not None}

        # Resource nodes, with the miner placed on each.
        self.text("GISEMENTS", (inner.x, inner.y), "small", MUTED)
        y = inner.y + 20
        for resource, purities in data.START_NODES.items():
            needed = data.RAW_RESOURCES[resource]
            locked = needed is not None and needed not in frame.unlocked
            self.text(RESOURCE_LABELS[resource], (inner.x, y + 18), "bold", DIM if locked else TEXT)
            for node, purity in enumerate(purities):
                tile = pygame.Rect(inner.x + 76 + node * 127, y, 121, 56)
                self.draw_node(tile, resource, purity, miners.get((resource, node)), locked)
            y += 64

        # Processing machines, grouped by recipe.
        y += 6
        self.text("MACHINES", (inner.x, y), "small", MUTED)
        y += 20
        groups: dict[tuple[str, str], list[MachineView]] = {}
        for machine in frame.machines:
            if machine.node is None:
                groups.setdefault((machine.building, machine.target), []).append(machine)
        if not groups:
            self.text("Aucune machine pour l'instant : tout se fait à la main.", (inner.x, y + 6), "body", DIM)
        for index, ((building, target), machines) in enumerate(sorted(groups.items())):
            col, row = index % 4, index // 4
            card = pygame.Rect(inner.x + col * 147, y + row * 104, 139, 96)
            self.draw_machine_group(card, building, target, machines)
        y += 104 * max(1, math.ceil(len(groups) / 4)) + 4

        # Power and logistics.
        y = max(y, inner.bottom - 64)
        self.text("ÉNERGIE ET LOGISTIQUE", (inner.x, y), "small", MUTED)
        y += 20
        if frame.power_capacity_mw:
            share = frame.power_used_mw / frame.power_capacity_mw
            self.text("Brûleur du HUB", (inner.x, y), "body", TEXT)
            self.bar(pygame.Rect(inner.x + 120, y + 4, 200, 12), share, TEAL)
            self.text(
                f"{frame.power_used_mw:4.1f} / {frame.power_capacity_mw:.0f} MW", (inner.x + 330, y), "mono", TEXT
            )
            fuel = f"carburant : {frame.fuel_seconds:.0f} s" if frame.power_demand_mw else "carburant : -"
            low = frame.power_demand_mw and frame.fuel_seconds < 30
            self.text(fuel, (inner.right, y), "small", BAD if low else MUTED, anchor="topright")
        else:
            self.text("Pas encore d'électricité (HUB Upgrade 2)", (inner.x, y), "body", DIM)
        y += 22
        if "conveyor_belt" in frame.unlocked:
            self.text("Convoyeurs : les machines sont reliées au stock", (inner.x, y), "body", TEAL)
        else:
            self.text("Pas de convoyeurs : il faut vider et remplir les machines à la main", (inner.x, y), "body", MUTED)

    def draw_node(self, tile: pygame.Rect, resource: str, purity: str, miner: MachineView | None, locked: bool) -> None:
        pygame.draw.rect(self.canvas, PANEL_2 if not locked else PANEL, tile, border_radius=5)
        pygame.draw.rect(self.canvas, LINE, tile, 1, border_radius=5)
        color = ITEM_COLORS[resource] if not locked else DIM
        pygame.draw.rect(self.canvas, color, (tile.x, tile.y + 6, 4, tile.height - 12), border_radius=2)
        self.text(PURITY_LABELS[purity], (tile.right - 8, tile.bottom - 18), "small", DIM, anchor="topright")
        if locked or miner is None:
            self.text("verrouillé" if locked else "libre", (tile.x + 12, tile.y + 6), "small", DIM)
            return
        name = data.BUILDINGS[miner.building].label
        self.text(name, (tile.x + 12, tile.y + 5), "small", TEXT, width=tile.width - 16)
        self.dot((tile.x + 16, tile.y + 28), STATUS[miner.status][1], 4)
        bar = pygame.Rect(tile.x + 25, tile.y + 24, tile.width - 34, 8)
        if miner.output_fill is not None:
            self.bar(bar, miner.output_fill, color, striped=miner.status == RUNNING)
            full = miner.output_fill > 0.5
            self.text("à vider" if full else "se remplit", (tile.x + 12, tile.bottom - 18), "small",
                      WARN if full else MUTED)
        else:
            self.bar(bar, miner.activity, TEAL, striped=miner.status == RUNNING)
            self.text("convoyeur", (tile.x + 12, tile.bottom - 18), "small", TEAL)

    def draw_machine_group(self, card: pygame.Rect, building: str, target: str, machines: list[MachineView]) -> None:
        pygame.draw.rect(self.canvas, PANEL_2, card, border_radius=5)
        running = sum(1 for m in machines if m.status == RUNNING)
        self.draw_icon(building, (card.x + 18, card.y + 20), running > 0)
        label = data.BUILDINGS[building].label
        self.text(f"{label} ×{len(machines)}" if len(machines) > 1 else label, (card.x + 36, card.y + 6), "bold", TEXT,
                  width=card.width - 40)
        outputs = data.RECIPES[target].outputs
        self.text("→ " + ", ".join(data.label(o) for o in outputs), (card.x + 36, card.y + 24), "small",
                  ITEM_COLORS.get(next(iter(outputs)), MUTED), width=card.width - 40)
        worst = min(machines, key=lambda m: (m.status == RUNNING, m.activity))
        status, color = STATUS[RUNNING if running == len(machines) else worst.status]
        self.dot((card.x + 12, card.y + 50), color, 4)
        self.text(status if running in (0, len(machines)) else f"{running}/{len(machines)} en marche",
                  (card.x + 22, card.y + 43), "small", color, width=card.width - 26)
        activity = sum(m.activity for m in machines) / len(machines)
        if machines[0].connected:
            self.text(f"cadence {activity:.0%}", (card.x + 10, card.y + 62), "small", TEAL)
            self.bar(pygame.Rect(card.x + 10, card.y + 79, card.width - 20, 8), activity, TEAL, striped=running > 0)
        else:
            fills_in = [m.input_fill or 0.0 for m in machines]
            fills_out = [m.output_fill or 0.0 for m in machines]
            half = (card.width - 26) // 2
            self.text("entrée", (card.x + 10, card.y + 62), "small", MUTED)
            self.text("sortie", (card.x + 16 + half, card.y + 62), "small", MUTED)
            self.bar(pygame.Rect(card.x + 10, card.y + 79, half, 8), sum(fills_in) / len(machines), MUTED)
            self.bar(pygame.Rect(card.x + 16 + half, card.y + 79, half, 8), sum(fills_out) / len(machines),
                     ITEM_COLORS.get(next(iter(outputs)), ACCENT), striped=running > 0)

    def draw_icon(self, building: str, center: tuple[int, int], active: bool) -> None:
        x, y = center
        color = ACCENT if active else MUTED
        if building == "smelter":  # furnace with a flame
            pygame.draw.rect(self.canvas, color, (x - 9, y - 8, 18, 16), 2, border_radius=3)
            if active:
                flicker = 2 * math.sin(self.anim * 12)
                pygame.draw.polygon(self.canvas, ACCENT, [(x - 4, y + 5), (x + 4, y + 5), (x, y - 4 - flicker)])
        else:  # constructor: a turning gear
            angle = self.anim * 3 if active else 0.0
            for k in range(6):
                a = angle + k * math.pi / 3
                pygame.draw.line(self.canvas, color, (x, y), (x + 10 * math.cos(a), y + 10 * math.sin(a)), 3)
            pygame.draw.circle(self.canvas, PANEL_2, (x, y), 6)
            pygame.draw.circle(self.canvas, color, (x, y), 6, 2)

    def draw_pioneer(self, rect: pygame.Rect, run: Run, frame: Frame) -> None:
        inner = self.panel(rect, "Pionnier")
        step = run.steps[run.step_at(self.time)] if run.steps else None
        finished = frame.upgrades_done >= len(data.HUB_UPGRADES)
        self.text("ACTION EN COURS", (inner.x, inner.y), "small", MUTED)
        if step is None or finished or self.time >= step.start + step.duration + 1e-6:
            label, share = ("Tier 0 terminé !" if finished else "Temps écoulé"), 1.0
        else:
            label = step.action.label()
            share = (self.time - step.start) / step.duration if step.duration else 1.0
        words, lines = label.split(), [""]
        for word in words:  # wrap on two lines
            trial = f"{lines[-1]} {word}".strip()
            if self.fonts["big"].size(trial)[0] > inner.width and lines[-1]:
                lines.append(word)
            else:
                lines[-1] = trial
        for k, line in enumerate(lines[:2]):
            self.text(line, (inner.x, inner.y + 18 + k * 24), "big", ACCENT if finished else TEXT, width=inner.width)
        self.bar(pygame.Rect(inner.x, inner.y + 70, inner.width, 8), share, ACCENT)

        y = inner.y + 92
        self.text("STOCK", (inner.x, y), "small", MUTED)
        y += 18
        column = inner.width // 2
        for index, item in enumerate(data.ITEMS):
            x = inner.x + (index % 2) * column
            row_y = y + (index // 2) * 20
            qty = frame.stock[item]
            color = TEXT if qty >= 1 else DIM
            pygame.draw.rect(self.canvas, ITEM_COLORS[item], (x, row_y + 4, 9, 9), border_radius=2)
            self.text(data.label(item), (x + 14, row_y), "small", color, width=column - 48)
            self.text(f"{qty:.0f}", (x + column - 6, row_y), "mono", color, anchor="topright")
        y += 20 * math.ceil(len(data.ITEMS) / 2) + 10

        self.text("JOURNAL", (inner.x, y), "small", MUTED)
        y += 18
        for start, action, count in self._recent_steps(run, rows=(inner.bottom - y) // 18):
            if action.kind == "submit":
                line, color = f"{short_clock(start)}  {self._upgrade_name(run, start)} livré", ACCENT
            else:
                times = f" ×{count}" if count > 1 else ""
                line, color = f"{short_clock(start)}  {action.label()}{times}", MUTED
            self.text(line, (inner.x, y), "small", color, width=inner.width)
            y += 18

    def _recent_steps(self, run: Run, rows: int) -> list:
        """Last actions started before now, repeated ones grouped, newest first."""
        if not run.steps:
            return []
        last = run.step_at(self.time)
        groups: list[list] = []
        index = last
        while index >= 0 and len(groups) < rows:
            step = run.steps[index]
            if groups and groups[-1][1] == step.action and step.action.kind != "submit":
                groups[-1][0] = step.start
                groups[-1][2] += 1
            else:
                groups.append([step.start, step.action, 1])
            index -= 1
        return groups

    @staticmethod
    def _upgrade_name(run: Run, start: float) -> str:
        done = sum(1 for t in run.upgrade_times if t <= start + data.SUBMIT_SECONDS + 1e-6)
        return data.HUB_UPGRADES[max(0, done - 1)].name

    def draw_race(self, rect: pygame.Rect, current: Run) -> None:
        inner = self.panel(rect, "La course")
        total = len(data.HUB_UPGRADES)
        self._race_rows = []
        y = inner.y
        for index, run in enumerate(self.runs[:4]):
            row = pygame.Rect(inner.x - 6, y - 2, inner.width + 12, 20)
            self._race_rows.append(row)
            selected = run is current
            if selected:
                pygame.draw.rect(self.canvas, PANEL_2, row, border_radius=4)
            self.text(f"{index + 1}  {run.name}", (inner.x, y), "bold" if selected else "body",
                      TEXT if selected else MUTED, width=130)
            bar = pygame.Rect(inner.x + 140, y + 4, inner.width - 400, 10)
            progress = run.progress_at(self.time)
            self.bar(bar, progress / total, ACCENT if selected else TEAL)
            for k in range(1, total):
                x = bar.x + bar.width * k / total
                pygame.draw.line(self.canvas, BG, (x, bar.y), (x, bar.bottom), 2)
            done = run.frame_at(self.time).upgrades_done
            if done >= total:
                status, color = f"Tier 0 fini en {short_clock(run.upgrade_times[-1])}", OK
            elif self.time >= run.end_time:
                status, color = f"{done}/{total} upgrades, abandon à {short_clock(run.end_time)}", BAD
            else:
                status, color = f"{done}/{total} upgrades · upgrade {done + 1} en cours", MUTED
            self.text(status, (bar.right + 14, y), "small", color, width=inner.right - bar.right - 14)
            y += 22
        # Timeline of the whole replay: click to jump.
        line = pygame.Rect(inner.x + 140, inner.bottom - 4, inner.width - 400, 6)
        self._timeline_rect = line
        self.text("Temps", (inner.x, line.y - 6), "small", MUTED)
        pygame.draw.rect(self.canvas, PANEL_2, line, border_radius=3)
        for split in current.upgrade_times:
            x = line.x + line.width * split / self.end_time
            pygame.draw.line(self.canvas, ACCENT, (x, line.y - 4), (x, line.bottom + 4), 2)
        x = line.x + line.width * self.time / self.end_time
        pygame.draw.circle(self.canvas, TEXT, (x, line.centery), 7)
        self.text(f"{short_clock(self.time)} / {short_clock(self.end_time)}", (line.right + 14, line.y - 6),
                  "mono", MUTED)
        self.text(HELP, (rect.right - 14, rect.y + 10), "small", MUTED, anchor="topright", width=640)


def show(runs: list[Run], start: int = 0, speed: int = 60, reference: str | None = "usine") -> None:
    """Open the window and play the runs until it is closed."""
    pygame.display.init()
    pygame.font.init()
    try:
        Viewer(runs, start, speed, reference).run()
    finally:
        pygame.quit()


def capture(runs: list[Run], path: str, time: float, current: int = 0, reference: str | None = "usine") -> None:
    """Save one image of the screen at a given game time, without opening a window."""
    pygame.font.init()
    viewer = Viewer(runs, current, reference=reference)
    viewer.time = min(time, viewer.end_time)
    viewer.anim = 0.3
    viewer.draw()
    pygame.image.save(viewer.canvas, path)
