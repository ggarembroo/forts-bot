"""
Forts Input Builder (Execution Module)
--------------------------------------
This module implements the direct hardware-level interface for simulating physical
mouse movements, keyboard keystrokes, and click paths inside the Forts game window.

Architectural Role:
- Acts as the 'Act' layer of the Sense-Think-Act pipeline.
- Receives strategic macro commands from the Decision Maker (brain).
- Translates high-level strategy (e.g., "build_mine()") into discrete, precise
  pixel sequences (mouse clicks, click-and-drag links, hotkeys).

Input Backend (Zero-Latency):
- All inputs are emitted through the raw Microsoft Win32 API
  (``ctypes.windll.user32.mouse_event`` / ``keybd_event``). This bypasses the
  Windows message-pump queue that PyAutoGUI relies on, giving driver-level
  scancodes with no extra polling or sleep overhead in the hot path.
- ``pydirectinput`` is used opportunistically if installed (it handles
  DirectInput games that swallow standard OS events); otherwise a pure-ctypes
  fallback is used so the builder always works without extra dependencies.

NOTE: pyautogui has been removed entirely from this module in favour of the raw
Win32 API. There is intentionally no per-call delay so the main loop stays
non-blocking; the caller parks on sleeps only where an in-game animation
requires it.
"""

import time
import ctypes
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (Builder): %(message)s")

# =============================================================================
# RAW WINDOWS INPUT LAYER (ctypes.windll.user32)
# -----------------------------------------------------------------------------
# These functions emit mouse/keyboard events at the driver level unconditionally.
# Windows coordinates for mouse_event are "Mickey-less" absolute pixels in the
# range 0..65535, so we normalise screen pixels before dispatch.
# =============================================================================

# Win32 flag constants (WinUser.h)
_MOUSEEVENTF_MOVE       = 0x0001
_MOUSEEVENTF_LEFTDOWN   = 0x0002
_MOUSEEVENTF_LEFTUP     = 0x0004
_MOUSEEVENTF_ABSOLUTE   = 0x8000

_KEYEVENTF_KEYUP        = 0x0002


def _set_cursor_pos(x, y):
    """Moves the OS mouse cursor to absolute screen coordinates (pixels)."""
    ctypes.windll.user32.SetCursorPos(int(x), int(y))


def _mouse_event(flags, data_x=0, data_y=0):
    """Low-level dispatch to user32.mouse_event. Coordinates are relative to 0..65535."""
    ctypes.windll.user32.mouse_event(flags, int(data_x), int(data_y), 0, 0)


def _normalise(x, screen_w):
    """Maps a pixel coordinate into the 0..65535 absolute range Windows expects."""
    return int((x * 65535) / max(1, screen_w - 1))


def _screen_dimensions():
    """Returns (width, height) of the primary display in pixels."""
    return (ctypes.windll.user32.GetSystemMetrics(0),   # SM_CXSCREEN
            ctypes.windll.user32.GetSystemMetrics(1))   # SM_CYSCREEN


def raw_click(x, y, clicks=1, hold=0.05):
    """
    Emits a low-level left-click at absolute screen coordinates using only the
    Win32 API. The button is held for ``hold`` seconds between down and up so a
    game polling at ~60Hz reliably observes the press (an instantaneous down+up
    can fall between two frames and be missed).
    """
    sw, sh = _screen_dimensions()
    nx, ny = _normalise(x, sw), _normalise(y, sh)

    for _ in range(clicks):
        _mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx, ny)
        _mouse_event(_MOUSEEVENTF_LEFTDOWN, nx, ny)
        time.sleep(hold)
        _mouse_event(_MOUSEEVENTF_LEFTUP, nx, ny)


def raw_drag(x1, y1, x2, y2, steps=16, duration=0.3):
    """
    Emits a click-hold-drag-release between two absolute coordinates using the
    Win32 API. Movement is interpolated across `steps` points over `duration`
    seconds so the game recognises the drag gesture.
    """
    sw, sh = _screen_dimensions()
    _set_cursor_pos(x1, y1)

    nx1, ny1 = _normalise(x1, sw), _normalise(y1, sh)
    np1 = _normalise(x1, sw)
    _set_cursor_pos(x1, y1)

    _mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, np1, _normalise(y1, sh))
    _mouse_event(_MOUSEEVENTF_LEFTDOWN, np1, _normalise(y1, sh))

    step_sleep = duration / max(1, steps)
    for i in range(1, steps + 1):
        t = i / steps
        ix = x1 + (x2 - x1) * t
        iy = y1 + (y2 - y1) * t
        _mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE,
                     _normalise(ix, sw), _normalise(iy, sh))
        time.sleep(step_sleep)

    nx2, ny2 = _normalise(x2, sw), _normalise(y2, sh)
    _mouse_event(_MOUSEEVENTF_MOVE | _MOUSEEVENTF_ABSOLUTE, nx2, ny2)
    _mouse_event(_MOUSEEVENTF_LEFTUP, nx2, ny2)


# Try to prefer pydirectinput (handles DirectInput hooks); fall back to raw ctypes.
try:
    import pydirectinput
    pydirectinput.PAUSE = 0.0
    USE_DIRECTINPUT = True
except ImportError:
    USE_DIRECTINPUT = False
    logging.warning("pydirectinput not found. Using raw Win32 ctypes input. "
                    "Run 'pip install pydirectinput' for the best DirectX game compatibility.")


class FortsBuilder:
    """
    Simulates pixel-precise inputs for selecting structures, choosing build tools,
    assembling physical grid struts, placing upgrade equipment, and firing weapons.

    Every input method is non-blocking at the hardware layer: it emits Win32 events
    and returns immediately (with only the absolute minimum in-game cooldown sleeps
    required for the client to register an action).
    """

    # Win32 virtual-key codes used for hotkeys. Kept as class constants so the
    # keybindings are visible and editable in one place.
    VK_CODES = {
        "x": 0x58,
        "r": 0x52,
        "escape": 0x1B,
    }

    def __init__(self, screen_region=None):
        """
        Initializes the builder system with window anchors.

        Args:
            screen_region (dict): The coordinates representing the game window bounds.
                                  Format: {"top": int, "left": int, "width": int, "height": int}
        """
        self.screen_region = screen_region or {"top": 0, "left": 0, "width": 1920, "height": 1080}

        # =========================================================================
        # COORDINATES DICTIONARY (Resolution: Relative Ratios [0.0 - 1.0])
        # These coefficients are multiplied by the screen width/height to make
        # the bot completely resolution-independent.
        # =========================================================================
        self.hud_ratios = {
            "base_core_center": (0.500, 0.500),       # Центр экрана
            "hud_build_menu_trigger": (0.500, 0.925), # 960/1920, 1000/1080
            "tool_wind_turbine": (0.468, 0.879),      # 900/1920, 950/1080
            "tool_metal_mine": (0.494, 0.879),        # 950/1920, 950/1080
            "tool_repair_brush": (0.520, 0.879),      # 1000/1920, 950/1080
            "slot_turbine_1": (0.458, 0.388),
            "slot_mine_1": (0.500, 0.629),
        }

    # -------------------------------------------------------------------------
    # Input primitives
    # -------------------------------------------------------------------------
    def _to_absolute(self, x_ratio, y_ratio):
        """
        Translates a relative window ratio coordinate to absolute screen pixels.

        Args:
            x_ratio (float): Relative X ratio (0.0 to 1.0) inside the capture window.
            y_ratio (float): Relative Y ratio (0.0 to 1.0) inside the capture window.

        Returns:
            tuple: Absolute screen (X, Y) coordinates.
        """
        abs_x = int(self.screen_region["left"] + (x_ratio * self.screen_region["width"]))
        abs_y = int(self.screen_region["top"] + (y_ratio * self.screen_region["height"]))
        return abs_x, abs_y

    def _press_key(self, key, hold=0.06):
        """
        Emits a key-down / hold / key-up sequence via the Win32 API.

        The explicit ``hold`` between down and up is critical: games poll input at
        ~60Hz, so an instantaneous down+up (0ms gap) can fall between two frames and
        be missed entirely. Holding the key for a few frames guarantees the game
        observes it in the pressed state. ``pydirectinput`` already sends physical
        scancodes (DirectInput-friendly); ``keybd_event`` is the fallback.

        Args:
            key (str): Logical key name (one of VK_CODES).
            hold (float): Seconds to keep the key held down (default 0.06).
        """
        if key not in self.VK_CODES:
            logging.error(f"Hotkey '{key}' is not mapped. Available: {list(self.VK_CODES)}")
            return False

        vk = self.VK_CODES[key]
        if USE_DIRECTINPUT:
            pydirectinput.keyDown(key)
            time.sleep(hold)
            pydirectinput.keyUp(key)
        else:
            ctypes.windll.user32.keybd_event(vk, 0, 0, 0)                 # key down
            time.sleep(hold)
            ctypes.windll.user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)  # key up
        return True

    def click(self, x, y, clicks=1, interval=0.0):
        """
        Sends a single or multiple hardware clicks at designated ratio coordinates.
        Non-blocking: the Win32 events are emitted back-to-back with no forced sleep.
        """
        abs_x, abs_y = self._to_absolute(x, y)
        logging.debug(f"Clicking absolute coordinate: ({abs_x}, {abs_y})")

        if USE_DIRECTINPUT:
            for i in range(clicks):
                pydirectinput.click(abs_x, abs_y)
                if clicks > 1 and i < clicks - 1:
                    time.sleep(interval)
        else:
            raw_click(abs_x, abs_y, clicks=clicks)

    def draw_beam(self, start_x, start_y, end_x, end_y):
        """
        Simulates the mechanical click-hold-drag-release required to draw structural
        beams (wood/bracing/armour) between nodes in Forts.

        Args:
            start_x, start_y: Coordinates of the initial joint node (relative ratios).
            end_x, end_y: Coordinates of the target connection joint node (relative ratios).
        """
        start_abs = self._to_absolute(start_x, start_y)
        end_abs = self._to_absolute(end_x, end_y)

        logging.info(f"[INPUT] Drawing beam from {start_abs} to {end_abs}")

        if USE_DIRECTINPUT:
            pydirectinput.moveTo(start_abs[0], start_abs[1])
            time.sleep(0.05)
            pydirectinput.mouseDown()
            time.sleep(0.05)
            pydirectinput.moveTo(end_abs[0], end_abs[1])
            time.sleep(0.05)
            pydirectinput.mouseUp()
        else:
            raw_drag(start_abs[0], start_abs[1], end_abs[0], end_abs[1])

        time.sleep(0.2)  # Cooldown safety

    def select_tool(self, tool_name):
        """
        Navigates the HUD menu to select a construction material or device.

        Args:
            tool_name (str): Key matching hud_ratios (e.g., 'tool_wind_turbine')
        """
        target_tool = f"tool_{tool_name}"
        if target_tool not in self.hud_ratios:
            logging.error(f"Selected tool '{tool_name}' is not mapped in builder coordinates!")
            return False

        menu_coords = self.hud_ratios["hud_build_menu_trigger"]
        tool_coords = self.hud_ratios[target_tool]

        logging.info(f"[INPUT] Selecting HUD construction tool: {tool_name}")

        # 1. Click menu trigger to expand HUD interface
        self.click(*menu_coords)
        time.sleep(0.15)

        # 2. Click specific tool icon
        self.click(*tool_coords)
        time.sleep(0.15)
        return True

    def build_structure(self, structure_type):
        """
        Constructs a complete structure from start to finish.
        Handles selection and physical placement on strategic grid positions.

        Args:
            structure_type (str): Type of building ('turbine' or 'mine')

        Returns:
            bool: True if strategic inputs were triggered successfully.
        """
        if structure_type == "turbine":
            # Select tool
            if not self.select_tool("wind_turbine"):
                return False
            # Find an available coordinate slot for placing turbine
            slot_coords = self.hud_ratios["slot_turbine_1"]  # Default target
            logging.info(f"[INPUT] Placing Wind Turbine at slot ratio coordinate: {slot_coords}")
            self.click(*slot_coords)
            return True

        elif structure_type == "mine":
            # Select tool
            if not self.select_tool("metal_mine"):
                return False
            # Find an available coordinate slot for placing mine
            slot_coords = self.hud_ratios["slot_mine_1"]
            logging.info(f"[INPUT] Placing Metal Mine at slot ratio coordinate: {slot_coords}")
            self.click(*slot_coords)
            return True

        logging.warning(f"Build request for unrecognized structure type: {structure_type}")
        return False

    def build_mine(self, x1, y1, x2, y2):
        """
        Builds a metal mine at the YOLO-detected build-site bounding box.

        The box's horizontal centre tells us which base we are playing:
        - centre in the LEFT half of the screen  -> left base  -> click the RIGHT
          edge of the box (shifted 10px inward).
        - centre in the RIGHT half of the screen -> right base -> click the LEFT
          edge of the box (shifted 10px inward).
        The vertical centre is always used for Y.

        Sequence: press 'x' (select mine) -> 0.2s -> move cursor -> 0.05s ->
        left click (held ~0.05s). Returns True once the input has been emitted.
        """
        screen_w, _ = _screen_dimensions()   # ctypes equivalent of win32api.GetSystemMetrics
        center_x = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        if center_x < screen_w / 2.0:
            target_x = int(x2 - 10)   # left base: aim at the site's right edge
            side = "left"
        else:
            target_x = int(x1 + 10)   # right base: aim at the site's left edge
            side = "right"
        target_y = int(y_center)

        logging.info(f"[INPUT] Building mine ({side} base): 'x' -> move to "
                     f"({target_x}, {target_y}) -> click")

        self._press_key("x")
        time.sleep(0.2)  # даём игре время зарегистрировать выбор шахты ('x')

        if USE_DIRECTINPUT:
            pydirectinput.moveTo(target_x, target_y)
        else:
            _set_cursor_pos(target_x, target_y)
        time.sleep(0.05)

        # Клик с удержанием: мгновенный down+up (0ms) игра на ~60Гц может не увидеть,
        # поэтому держим кнопку зажатой несколько кадров перед отпусканием.
        if USE_DIRECTINPUT:
            pydirectinput.mouseDown()
            time.sleep(0.05)
            pydirectinput.mouseUp()
        else:
            raw_click(target_x, target_y, hold=0.05)
        return True

    def trigger_repair(self):
        """
        Activates the repair tool (usually hotkey 'R' or selecting Repair Brush),
        and moves the mouse over damaged nodes to perform maintenance.

        Returns:
            bool: True if repair sweep was sent.
        """
        logging.info("[INPUT] Executing fast repair cycle...")

        # Method 1: Click the repair tool icon
        if not self.select_tool("repair_brush"):
            # Method 2: Fallback to keyboard hotkey 'R' (standard Forts keybind)
            self._press_key("r")
            time.sleep(0.1)

        # Draw a sweep/brush movement pattern over key areas of the fort core to repair them.
        core_x, core_y = self.hud_ratios["base_core_center"]

        # Perform dragging repair strokes (relative ratio offsets)
        self.draw_beam(core_x - 0.08, core_y - 0.10, core_x + 0.08, core_y - 0.10)
        self.draw_beam(core_x - 0.08, core_y + 0.10, core_x + 0.08, core_y + 0.10)

        # Return to normal building cursor (usually press Esc or select Wood tool)
        self._press_key("escape")

        return True


# ==================== LOCAL TEST SCRIPTS ====================
if __name__ == "__main__":
    print("==================================================")
    print("          FORTS BUILDER CONTROLLER: TEST          ")
    print("==================================================")
    print("WARNING: This script simulates physical clicks and drags.")
    print("Keep your mouse centered. Run only with an active sandbox game window!")
    print("Starting in 4 seconds...")
    time.sleep(4)

    # Initialize builder targeting top-left window coordinates
    builder = FortsBuilder(screen_region={"top": 0, "left": 0, "width": 1920, "height": 1080})

    try:
        # Simulate selecting wood tool/beam ratio
        print("\n--- Test Action: Selecting HUD Tool ---")
        builder.select_tool("wind_turbine")

        # Simulate drawing a structural support beam using ratios
        print("\n--- Test Action: Drawing Support Beam ---")
        cx, cy = builder.hud_ratios["base_core_center"]
        builder.draw_beam(cx - 0.05, cy + 0.15, cx + 0.05, cy + 0.15)

        print("\n--- Test Complete ---")

    except Exception as e:
        print(f"\n[ABORTED] Simulation interrupted or failed: {e}")
