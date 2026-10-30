"""Staying alive: face a creature and strike it, run away, get back up after dying.

Health doesn't come back by itself in the game: resting is useless, fighting well matters."""
from __future__ import annotations

from config import KEYS
from game_state import read_player
from moves import Mover, SkillError

STRIKE_SIZE = 0.22  # a creature this big on screen is within reach of the Xeno-Zapper
POINT_BLANK = 0.35  # this big: about to hit, strike whatever it is
APPROACHING = 1.2  # grown this much in about a second: it is coming at us
FIGHT_CONFIDENCE = 0.08  # during a fight, even a doubtful shape is worth facing
FLEE_HP = 35  # below this health while still being hit: run
BACK_OFF_S = 0.35  # step back this long after striking: a charging creature misses
RUSH_AFTER_S = 1.5  # hurt by a creature that stays out of reach this long: run at it
RUSH_HP = 50  # ...if health is at least this
FIGHT_S = 45  # longest fight before running away
CALM_S = 4  # no damage for this long: whatever is around is not a danger
HURT_MEMORY_S = 5  # still fighting back this long after the last hit
IGNORE_HARMLESS_S = 20  # don't stop again for a harmless creature during this time
FLEE_S = 8
RESPAWN_WORDS = ("Réapparaître", "Réapparition", "Respawn", "Revivre")


def defend(game, mover: Mover, eyes, home: tuple[float, float] | None, hurt: bool, log=print,
           charging: bool = False) -> str:
    """Deal with a creature or an attack. Returns "safe", "harmless" or "fled".

    Not every creature is hostile: strike only when losing health, when the
    creature rushes at the player, or when it is right in front. Fight first;
    run only when health is low and the hits keep coming.
    """
    game.alarms = False  # the fight itself must not be interrupted by the alarm
    confidence = eyes.confidence if eyes else None
    if eyes:
        eyes.confidence = FIGHT_CONFIDENCE
    try:
        start = calm_since = game.now()
        hurt_at = start if hurt else -1e9
        charged_at = start if charging else -1e9  # a creature was coming at us
        last_hp = game.player().hp
        sizes: list[tuple[float, float]] = []  # (time, size) of the creature watched
        looks = 0
        out_of_reach_since = None  # a creature hurting us from afar: since when
        mover.set_pitch(5)
        while game.now() - start < FIGHT_S:
            now = game.now()
            player = game.player()
            if player.hp < last_hp - 0.5:
                hurt_at = calm_since = now
            last_hp = player.hp
            if player.hp < FLEE_HP and now - hurt_at < 3:
                flee(game, mover, home, log)
                return "fled"
            view = eyes.fresh(now - 0.1, timeout=2.0) if eyes else None
            threat = view.nearest_threat() if view else None
            quiet = now - hurt_at > CALM_S
            if threat is None:
                if quiet and now - calm_since > CALM_S:
                    return "safe"
                if now - hurt_at < 1.0:
                    game.click()  # just hit: the attacker may be right in front, unseen
                if looks == 0:
                    log("  je cherche l'attaquant")
                mover.turn(180 if looks % 3 == 0 else 90)  # attackers often come from behind
                looks += 1
                game.wait(0.25)
                continue
            sizes = [(t, s) for t, s in sizes if now - t < 1.5] + [(now, threat.size)]
            approaching = threat.size > APPROACHING * sizes[0][1] and now - sizes[0][0] > 0.5
            if quiet and not approaching and now - start > CALM_S:
                game.ignore_sightings_until = now + IGNORE_HARMLESS_S
                return "harmless"
            mover.turn(threat.bearing, speed=400)
            mover.tilt(threat.below, speed=400)
            hostile = (now - hurt_at < HURT_MEMORY_S or now - charged_at < HURT_MEMORY_S or approaching
                       or threat.size >= POINT_BLANK)
            if hostile and threat.size >= STRIKE_SIZE:
                log(f"  je frappe ({threat.label})")
                for _ in range(3):
                    game.click()
                    game.wait(0.25)
                back_off(game, mover)
                out_of_reach_since = None
            elif now - hurt_at < HURT_MEMORY_S and not approaching:  # it hurts us from afar
                out_of_reach_since = out_of_reach_since or now
                if now - out_of_reach_since > RUSH_AFTER_S and player.hp >= RUSH_HP:
                    rush(game, mover, eyes, log)
                    out_of_reach_since = None
                else:
                    game.wait(0.15)
            else:
                out_of_reach_since = None
                game.wait(0.15)  # watch it
        flee(game, mover, home, log)
        return "fled"
    finally:
        game.alarms = True
        if eyes:
            eyes.confidence = confidence


def back_off(game, mover: Mover) -> None:
    """Step back a little, unless the ground falls away behind."""
    player = game.player()
    if mover.terrain.edge_ahead(player.x, player.y, mover.facing(player) + 180):
        return
    game.hold(KEYS["back"])
    game.wait(BACK_OFF_S)
    game.release(KEYS["back"])


def rush(game, mover: Mover, eyes, log=print) -> None:
    """Run at a creature that keeps out of reach (one that spits from afar), to strike it."""
    player = game.player()
    if mover.terrain.edge_ahead(player.x, player.y, mover.facing(player), 4.0):
        return  # not over a cliff
    log("  elle m'attaque de loin : je fonce sur elle")
    game.hold(KEYS["sprint"])
    game.hold(KEYS["forward"])
    try:
        for _ in range(5):
            game.wait(0.2)
            view = getattr(eyes, "latest", None)
            threat = view.nearest_threat() if view else None
            if threat is None or threat.size >= STRIKE_SIZE:
                break
            mover.turn(threat.bearing, speed=500)
    finally:
        game.release(KEYS["forward"])
        game.release(KEYS["sprint"])


def flee(game, mover: Mover, home: tuple[float, float] | None, log=print) -> None:
    """Run towards home (the HUB) if known, along a safe way; else straight back, jumping now
    and then, stopping before a known edge."""
    log("  je fuis")
    if home is not None:
        if mover.walker is None:
            game.hold(KEYS["sprint"])
        try:
            mover.walk_to(*home, radius=5.0, timeout=FLEE_S)  # the learnt walk runs by itself
        finally:
            game.release(KEYS["sprint"])
        return
    mover.turn(180, speed=500)
    mover.set_pitch(5)
    game.hold(KEYS["sprint"])
    game.hold(KEYS["forward"])
    try:
        for step in range(int(FLEE_S / 0.5)):
            game.wait(0.5)
            player = game.player()
            if mover.terrain.edge_ahead(player.x, player.y, mover.facing(player), 4.0):
                break
            if step % 4 == 3:
                game.hold(KEYS["jump"])
                game.wait(0.1)
                game.release(KEYS["jump"])
    finally:
        game.release(KEYS["forward"])
        game.release(KEYS["sprint"])


def respawn(game, screen, log=print) -> None:
    """After death: right-click, as the game asks (« Appuyez sur Clic droit pour réapparaître »),
    and wait to be alive again. Now and then, click the message itself, in case it is a button."""
    game.alarms = False
    try:
        for attempt in range(20):
            if not read_player().dead:
                log("  de nouveau en vie")
                return
            game.click("right")
            if attempt % 4 == 3:
                for word in RESPAWN_WORDS:
                    button = screen.find(word, "center")
                    if button is not None:
                        game.click_at(button.x, button.y)
                        break
            game.wait(1.5)
        raise SkillError("je n'arrive pas à réapparaître")
    finally:
        game.alarms = True
