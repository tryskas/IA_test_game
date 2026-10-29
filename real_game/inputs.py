"""Keyboard and mouse for the game, through the Windows SendInput function.

Keys are sent as scan codes (the physical key), which DirectX games read.
A key is named by the letter the game shows in its controls ("z" for forward
on an AZERTY keyboard): the current keyboard layout tells which physical key
types that letter, so the same names work on AZERTY and QWERTY.
"""
from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes

if sys.platform != "win32":
    raise ImportError("inputs.py ne marche que sous Windows")

user32 = ctypes.WinDLL("user32", use_last_error=True)

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSE_BUTTONS = {"left": (0x0002, 0x0004), "right": (0x0008, 0x0010), "middle": (0x0020, 0x0040)}
MAPVK_VK_TO_VSC_EX = 4

# Virtual-key codes of keys that are not a character.
NAMED_KEYS = {
    "space": 0x20, "tab": 0x09, "enter": 0x0D, "escape": 0x1B, "backspace": 0x08,
    "shift": 0xA0, "ctrl": 0xA2, "alt": 0xA4,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    **{f"f{n}": 0x6F + n for n in range(1, 13)},
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]


# Same layout as the Windows header, or SendInput would refuse the inputs.
assert ctypes.sizeof(INPUT) == (40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)

user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT
user32.VkKeyScanW.argtypes = (wintypes.WCHAR,)
user32.VkKeyScanW.restype = ctypes.c_short
user32.MapVirtualKeyW.argtypes = (wintypes.UINT, wintypes.UINT)
user32.MapVirtualKeyW.restype = wintypes.UINT


def _send(*events: INPUT) -> None:
    array = (INPUT * len(events))(*events)
    if user32.SendInput(len(events), array, ctypes.sizeof(INPUT)) != len(events):
        raise ctypes.WinError(ctypes.get_last_error())


def scan_code(key: str) -> tuple[int, bool]:
    """Physical key for a key name: (scan code, is an extended key)."""
    key = key.lower()
    if key in NAMED_KEYS:
        vk = NAMED_KEYS[key]
    elif len(key) == 1:
        found = user32.VkKeyScanW(key)
        if found == -1:
            raise ValueError(f"aucune touche ne tape « {key} » avec ce clavier")
        vk = found & 0xFF
    else:
        raise ValueError(f"touche inconnue : {key}")
    code = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC_EX)
    return code & 0xFF, (code >> 8) in (0xE0, 0xE1)


def _key_event(key: str, up: bool) -> INPUT:
    scan, extended = scan_code(key)
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0) | (KEYEVENTF_KEYUP if up else 0)
    return INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=0, wScan=scan, dwFlags=flags))


def key_down(key: str) -> None:
    _send(_key_event(key, up=False))


def key_up(key: str) -> None:
    _send(_key_event(key, up=True))


def press(key: str, hold_s: float = 0.05) -> None:
    """Tap a key. Games check keys once per frame, so it stays down a short moment."""
    key_down(key)
    time.sleep(hold_s)
    key_up(key)


def move(dx: int, dy: int) -> None:
    """Move the mouse by (dx, dy): turns the camera in game."""
    _send(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dx=dx, dy=dy, dwFlags=MOUSEEVENTF_MOVE)))


def move_to(x: int, y: int) -> None:
    """Put the cursor at a point of the screen, for menus where the cursor shows."""
    user32.SetCursorPos(int(x), int(y))


def mouse_down(button: str = "left") -> None:
    _send(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dwFlags=MOUSE_BUTTONS[button][0])))


def mouse_up(button: str = "left") -> None:
    _send(INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(dwFlags=MOUSE_BUTTONS[button][1])))


def click(button: str = "left", hold_s: float = 0.05) -> None:
    mouse_down(button)
    time.sleep(hold_s)
    mouse_up(button)


KEYEVENTF_UNICODE = 0x0004


def type_text(text: str, delay_s: float = 0.03) -> None:
    """Type text as characters (accents included), whatever the keyboard layout."""
    for char in text:
        down = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=0, wScan=ord(char), dwFlags=KEYEVENTF_UNICODE))
        up = INPUT(type=INPUT_KEYBOARD,
                   ki=KEYBDINPUT(wVk=0, wScan=ord(char), dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
        _send(down, up)
        time.sleep(delay_s)


def double_click(button: str = "left") -> None:
    click(button)
    time.sleep(0.08)
    click(button)


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


def glide_to(x: float, y: float, seconds: float = 0.18) -> None:
    """Move the cursor to (x, y) smoothly, speeding up then slowing down, like a hand."""
    here = POINT()
    user32.GetCursorPos(ctypes.byref(here))
    steps = max(2, int(seconds / 0.012))
    for k in range(1, steps + 1):
        t = k / steps
        eased = t * t * (3 - 2 * t)
        user32.SetCursorPos(int(here.x + (x - here.x) * eased), int(here.y + (y - here.y) * eased))
        time.sleep(seconds / steps)
