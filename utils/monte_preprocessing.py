import numpy as np
from collections import namedtuple

MonteRAMXYScreen = namedtuple("MonteRAMXY", ["player_x", "player_y", "screen"])
MonteRAMState = namedtuple("MonteRAMState", ["player_x", "player_y", "screen", "has_key", "door_left_locked", "door_right_locked", "skull_x", "lives"])

def get_byte(ram: np.ndarray, address: int) -> int:
    """Return the byte at the specified emulator RAM location"""
    assert isinstance(ram, np.ndarray) and ram.dtype == np.uint8 and isinstance(address, int)
    return int(ram[address & 0x7f])

def parse_ram(ram: np.ndarray) -> MonteRAMState:
    """Get the current annotated Montezuma RAM state as a tuple

    See RAM annotations:
    https://docs.google.com/spreadsheets/d/1KU4KcPqUuhSZJ1N2IyhPW59yxsc4mI4oSiHDWA3HCK4
    """
    x = get_byte(ram, 0xaa)
    y = get_byte(ram, 0xab)
    screen = get_byte(ram, 0x83)

    inventory = get_byte(ram, 0xc1)
    key_mask = 0b00000010
    has_key = bool(inventory & key_mask)

    objects = get_byte(ram, 0xc2)
    door_left_locked  = bool(objects & 0b1000)
    door_right_locked = bool(objects & 0b0100)

    skull_offset = 33
    skull_x = get_byte(ram, 0xaf) + skull_offset
    #skull_x = 0
    lives = get_byte(ram, 0xba)

    return MonteRAMState(x, y, screen, has_key, door_left_locked, door_right_locked, skull_x, lives)

def parse_ram_xy_screen(ram: np.ndarray):
    """Get the current annotated Montezuma RAM state as a tuple

    See RAM annotations:
    https://docs.google.com/spreadsheets/d/1KU4KcPqUuhSZJ1N2IyhPW59yxsc4mI4oSiHDWA3HCK4
    """
    x = get_byte(ram, 0xaa)
    y = get_byte(ram, 0xab)
    screen = get_byte(ram, 0x83)

    return MonteRAMXYScreen(x, y, screen)

def parse_ram_xy(ram: np.ndarray):
    """Get the current annotated Montezuma RAM state as a tuple

    See RAM annotations:
    https://docs.google.com/spreadsheets/d/1KU4KcPqUuhSZJ1N2IyhPW59yxsc4mI4oSiHDWA3HCK4
    """
    x = get_byte(ram, 0xaa)
    y = get_byte(ram, 0xab)

    return x, y

def discretize_state(state: MonteRAMState, discretize_factor: int) -> MonteRAMState:
    x, y, has_key, door_left_locked, door_right_locked, skull_x, lives = state
    x //= discretize_factor
    y //= discretize_factor
    skull_x //= discretize_factor
    return MonteRAMState(
        x, y, has_key, door_left_locked, door_right_locked, skull_x, lives
    )
