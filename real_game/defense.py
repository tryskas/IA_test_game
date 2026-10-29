"""Staying alive: face a creature and strike it, run away, get back up after dying."""
from __future__ import annotations

from config import KEYS
from game_state import read_player
from moves import Mover, SkillError

STRIKE_SIZE = 0.22  # a creature this big on screen is within reach of the Xeno-Zapper
FLEE_HP = 35  # below this health, stop fighting and run
FIGHT_S = 45  # longest fight before running away
CALM_S = 4  # no creature seen and no damage for this long: safe again
FLEE_S = 8
RESPAWN_WORDS = ("Réapparaître", "Réapparition", "Respawn", "Revivre")


def defend(game, mover: Mover, eyes, home: tuple[float, float] | None, log=print) -> str:
    """Fight what attacks the player. Returns "safe" or "fled"."""
    game.alarms = False  # the fight itself must not be interrupted by the alarm
    try:
        start = calm_since = game.now()
        last_hp = game.player().hp
        mover.set_pitch(5)
        while game.now() - start < FIGHT_S:
            player = game.player()
            if player.hp < last_hp - 0.5:
                calm_since = game.now()
            last_hp = player.hp
            if player.hp < FLEE_HP:
                flee(game, mover, home, log)
                return "fled"
            view = eyes.fresh(game.now() - 0.1, timeout=2.0) if eyes else None
            threat = view.nearest_threat() if view else None
            if threat is None:
                if game.now() - calm_since > CALM_S:
                    return "safe"
                mover.turn(90)  # hit by something unseen: look around
                game.wait(0.3)
                continue
            calm_since = game.now()
            log(f"  créature ({threat.label}) à {threat.bearing:+.0f}°, je lui fais face")
            mover.turn(threat.bearing)
            game.mouse(0, round(threat.below / mover.cal.pitch_per_count))
            if threat.size >= STRIKE_SIZE:
                for _ in range(3):
                    game.click()
                    game.wait(0.25)
            else:
                game.wait(0.2)  # let it come within reach
        flee(game, mover, home, log)
        return "fled"
    finally:
        game.alarms = True


def flee(game, mover: Mover, home: tuple[float, float] | None, log=print) -> None:
    """Sprint towards home (the HUB) if known, else straight back, jumping now and then."""
    log("  je fuis")
    if home is not None:
        mover.face(*home)
    else:
        mover.turn(180)
    mover.set_pitch(5)
    game.hold(KEYS["sprint"])
    game.hold(KEYS["forward"])
    try:
        for step in range(int(FLEE_S / 0.5)):
            game.wait(0.5)
            if step % 4 == 3:
                game.hold(KEYS["jump"])
                game.wait(0.1)
                game.release(KEYS["jump"])
    finally:
        game.release(KEYS["forward"])
        game.release(KEYS["sprint"])


def respawn(game, screen, log=print) -> None:
    """After death: find the respawn button, click it, wait to be alive again."""
    game.alarms = False
    try:
        for _ in range(20):
            if not read_player().dead:
                log("  de nouveau en vie")
                return
            for word in RESPAWN_WORDS:
                button = screen.find(word, "center")
                if button is not None:
                    game.click_at(button.x, button.y)
                    break
            game.wait(1.5)
        raise SkillError("je n'arrive pas à réapparaître")
    finally:
        game.alarms = True
