"""Watch the AI walk across the simulated landscapes it trains on.

    python watch_walk.py                    # the trained walker (models/marche.zip), else the simple rule
    python watch_walk.py --joueur regle     # also: ia, hasard

Seen from above: hills and plateaus (green to brown), too steep to walk (dark),
ravines (blue). The 15 lines in front of the walker are what
its eyes report: red where something is in the way, orange dots where the ground
falls away. Purple squares are what its map remembers; red edges are where a fall hurts.

Keys: space pause, N new landscape, 1 AI, 2 simple rule, 3 random, + / - speed, Esc quit.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "real_game"))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from navigation import CELL  # noqa: E402
from navigation import DROP_AHEAD  # noqa: E402
from terrain_sim import (ACTIONS, BEARINGS, DT, GOAL_REACHED, SAFE_FALL, SIGHT, SIZE, STEP_UP,  # noqa: E402
                         WalkEnv, greedy_action)

SCALE = 7  # pixels per metre
MAP = SIZE * SCALE
PANEL = 330
PLAYERS = {"ia": "IA entraînée", "regle": "règle simple", "hasard": "au hasard"}
HEIGHT_COLOURS = [(-20, (55, 45, 130)), (-7, (60, 110, 200)), (-5, (110, 165, 95)), (0, (150, 200, 110)),
                  (4, (225, 215, 150)), (8, (190, 160, 120)), (14, (150, 130, 125))]
MEMORY = (190, 80, 230)
BACKGROUND, TEXT, DIM = (24, 26, 30), (235, 235, 235), (150, 155, 165)


def terrain_image(h: np.ndarray) -> pygame.Surface:
    """Colours by height, light from the top left, dark lines where it is too steep to walk,
    red on edges where falling hurts."""
    levels = [lv for lv, _ in HEIGHT_COLOURS]
    rgb = np.stack([np.interp(h, levels, [c[i] for _, c in HEIGHT_COLOURS]) for i in range(3)], axis=-1)
    gy, gx = np.gradient(h)
    light = np.clip(1.0 - 0.25 * (gx + gy), 0.6, 1.25)[..., None]
    rgb = rgb * light
    rgb[np.hypot(gx, gy) > STEP_UP] *= 0.35
    padded = np.pad(h, 1, mode="edge")
    lowest = np.minimum.reduce([padded[:-2, 1:-1], padded[2:, 1:-1], padded[1:-1, :-2], padded[1:-1, 2:]])
    rgb[h - lowest > SAFE_FALL] = (220, 40, 45)  # falling from here hurts
    image = np.clip(rgb, 0, 255).astype(np.uint8).transpose(1, 0, 2)  # pygame wants (x, y)
    return pygame.transform.scale(pygame.surfarray.make_surface(image), (MAP, MAP))


def screen(x: float, y: float) -> tuple[int, int]:
    return int(x * SCALE), int(y * SCALE)


class Viewer:
    def __init__(self, player: str, seed: int):
        pygame.init()
        pygame.display.set_caption("IA Satisfactory : apprendre à marcher")
        self.window = pygame.display.set_mode((MAP + PANEL, MAP))
        self.font = pygame.font.Font(None, 24)
        self.big = pygame.font.Font(None, 34)
        self.env = WalkEnv()
        self.walker = None
        self.player = self.choose(player)
        self.seed = seed
        self.steps_per_s = 10.0  # 5 = real time
        self.paused = False
        self.totals = {"essais": 0, "arrivées": 0, "chutes": 0, "morts": 0}
        self.rng = np.random.default_rng(seed)
        self.new_landscape()

    def choose(self, player: str) -> str:
        if player == "ia" and self.walker is None:
            from walk_policy import WalkPolicy

            self.walker = WalkPolicy.load()
            if self.walker is None:
                print("Pas encore de models/marche.zip (lance « python train_walk.py ») : règle simple.")
                return "regle"
        return player

    def new_landscape(self) -> None:
        self.obs, _ = self.env.reset(seed=self.seed)
        self.seed += 1
        self.background = terrain_image(self.env.h)
        self.trail = [(self.env.x, self.env.y)]
        self.start = (self.env.x, self.env.y)
        self.result, self.result_at, self.fall_at, self.action = None, 0.0, -10.0, 0

    def act(self) -> int:
        if self.player == "ia":
            return self.walker.act(self.obs)
        if self.player == "regle":
            return greedy_action(self.obs)
        return int(self.rng.integers(len(ACTIONS)))

    def step(self, now: float) -> None:
        self.action = self.act()
        self.obs, _, terminated, truncated, info = self.env.step(self.action)
        self.trail.append((self.env.x, self.env.y))
        if info["fell"]:
            self.fall_at = now
            self.totals["chutes"] += 1
        if terminated or truncated:
            self.totals["essais"] += 1
            if info["reached"]:
                self.totals["arrivées"] += 1
                self.result = ("Arrivé !", (90, 210, 120))
            elif info["dead"]:
                self.totals["morts"] += 1
                self.result = ("Mort (chute)", (230, 70, 70))
            else:
                self.result = ("Temps écoulé", (230, 180, 60))
            self.result_at = now

    # Drawing -------------------------------------------------------------------------

    def draw_map(self, now: float) -> None:
        env, win = self.env, self.window
        win.blit(self.background, (0, 0))
        memory = pygame.Surface((int(CELL * SCALE), int(CELL * SCALE)), pygame.SRCALPHA)
        memory.fill((*MEMORY, 120))
        for cell in env.map.blocked:
            if env.map.is_blocked(cell):
                win.blit(memory, screen(cell[0] * CELL, cell[1] * CELL))
        if len(self.trail) > 1:
            pygame.draw.lines(win, (40, 110, 240), False, [screen(*p) for p in self.trail], 3)
        pygame.draw.circle(win, (40, 170, 80), screen(*self.start), 7)
        goal = screen(*env.goal)
        pygame.draw.circle(win, (250, 200, 30), goal, int(GOAL_REACHED * SCALE), 3)
        pygame.draw.circle(win, (250, 200, 30), goal, 6)
        here = screen(env.x, env.y)
        distances, drops = env.sight if env.sight else ([None] * len(BEARINGS), [False] * len(BEARINGS))
        for bearing, distance, drop in zip(BEARINGS, distances, drops):
            angle = math.radians(env.facing + bearing)
            reach = SIGHT if distance is None else distance
            end = screen(env.x + reach * math.cos(angle), env.y + reach * math.sin(angle))
            pygame.draw.line(win, (255, 255, 255) if distance is None else (235, 50, 50), here, end,
                             1 if distance is None else 3)
            if drop:
                pygame.draw.circle(win, (255, 150, 30), screen(env.x + DROP_AHEAD * math.cos(angle),
                                                               env.y + DROP_AHEAD * math.sin(angle)), 6)
        facing = math.radians(env.facing)
        tip = (env.x + 1.8 * math.cos(facing), env.y + 1.8 * math.sin(facing))
        left = (env.x + 1.1 * math.cos(facing + 2.5), env.y + 1.1 * math.sin(facing + 2.5))
        right = (env.x + 1.1 * math.cos(facing - 2.5), env.y + 1.1 * math.sin(facing - 2.5))
        pygame.draw.polygon(win, (255, 255, 255), [screen(*tip), screen(*left), screen(*right)])
        pygame.draw.polygon(win, (20, 20, 20), [screen(*tip), screen(*left), screen(*right)], 2)
        if now - self.fall_at < 1.0:
            self.text("Chute !", (here[0] + 14, here[1] - 30), (255, 80, 60), self.big)

    def text(self, words: str, at, colour=TEXT, font=None) -> None:
        self.window.blit((font or self.font).render(words, True, colour), at)

    def draw_panel(self) -> None:
        env, x = self.env, MAP + 18
        pygame.draw.rect(self.window, BACKGROUND, (MAP, 0, PANEL, MAP))
        self.text(f"Marcheur : {PLAYERS[self.player]}", (x, 18), font=self.big)
        self.text(f"Paysage n° {self.seed - 1}    temps {env.steps * DT:5.1f} s", (x, 56))
        self.text(f"Vitesse : x{self.steps_per_s * DT:.0f}" + ("   (pause)" if self.paused else ""), (x, 80), DIM)
        self.text("Vie", (x, 112))
        pygame.draw.rect(self.window, (70, 70, 75), (x + 40, 114, 240, 16))
        pygame.draw.rect(self.window, (90, 210, 120) if env.hp > 50 else (230, 80, 60),
                         (x + 40, 114, int(240 * max(env.hp, 0) / 100), 16))
        self.text(f"Geste : {ACTIONS[self.action]}", (x, 142))
        self.text(f"But à {env.goal_distance():.0f} m", (x, 166))

        self.text("Ce que voient ses yeux :", (x, 204))
        distances, drops = env.sight if env.sight else ([None] * len(BEARINGS), [False] * len(BEARINGS))
        recalled = env.map.blocked_ahead(env.x, env.y, env.facing, BEARINGS)
        for i, (distance, drop, memory) in enumerate(zip(distances, drops, recalled)):
            box = pygame.Rect(x + i * 19, 230, 17, 26)
            colour = (60, 62, 68)
            if distance is not None:
                colour = (int(120 + 135 * (1 - distance / SIGHT)), 45, 45)
            pygame.draw.rect(self.window, colour, box)
            if drop:
                pygame.draw.rect(self.window, (255, 150, 30), box.inflate(-8, -12))
            if memory is not None:
                pygame.draw.rect(self.window, MEMORY, box, 2)
        self.text("gauche", (x, 260), DIM)
        self.text("droite", (x + 230, 260), DIM)

        if self.result:
            self.text(self.result[0], (x, 300), self.result[1], self.big)
        t = self.totals
        self.text(f"Essais {t['essais']}   arrivées {t['arrivées']}", (x, 346))
        self.text(f"Chutes {t['chutes']}   morts {t['morts']}", (x, 370))

        legend = [((40, 110, 240), "trajet"), ((250, 200, 30), "but"), ((235, 50, 50), "obstacle vu"),
                  ((255, 150, 30), "le sol se dérobe"), (MEMORY, "retenu sur sa carte"),
                  ((220, 40, 45), "bord : chute de plus de 4 m"), ((40, 40, 40), "trop raide pour marcher"),
                  ((60, 110, 200), "fond de ravin")]
        for k, (colour, words) in enumerate(legend):
            pygame.draw.rect(self.window, colour, (x, 420 + 24 * k, 16, 16))
            self.text(words, (x + 26, 419 + 24 * k), DIM)
        keys = ["Espace : pause      N : autre paysage", "1 : IA   2 : règle simple   3 : hasard",
                "+ / - : vitesse      Échap : quitter"]
        for k, words in enumerate(keys):
            self.text(words, (x, MAP - 84 + 24 * k), DIM)

    # Loop ----------------------------------------------------------------------------

    def run(self, frames: int | None = None) -> None:
        clock = pygame.time.Clock()
        owed = 0.0
        frame = 0
        while frames is None or frame < frames:
            frame += 1
            now = pygame.time.get_ticks() / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_n:
                        self.new_landscape()
                    elif event.unicode in ("1", "2", "3"):
                        self.player = self.choose(("ia", "regle", "hasard")[int(event.unicode) - 1])
                    elif event.unicode == "+":
                        self.steps_per_s = min(80.0, self.steps_per_s * 2)
                    elif event.unicode == "-":
                        self.steps_per_s = max(2.5, self.steps_per_s / 2)
            if self.result:
                if now - self.result_at > 1.5:
                    self.new_landscape()
            elif not self.paused:
                owed += self.steps_per_s / 30
                while owed >= 1 and not self.result:
                    owed -= 1
                    self.step(now)
            self.draw_map(now)
            self.draw_panel()
            pygame.display.flip()
            clock.tick(30)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--joueur", choices=list(PLAYERS), default="ia")
    parser.add_argument("--graine", type=int, default=10_000, help="premier paysage")
    args = parser.parse_args()
    Viewer(args.joueur, args.graine).run()
    pygame.quit()


if __name__ == "__main__":
    main()
