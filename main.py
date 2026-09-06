"""
FortsBot: Core Orchestrator (Entry Point)
-----------------------------------------
This file binds the Perception, Decision, and Execution modules together into a
unified, high-performance, non-blocking execution loop.

Architecture (Producer-Consumer, multiprocessing):
1. The main process owns a ``multiprocessing.Manager().dict()`` (``shared_state``)
   that is the single source of truth for the latest parsed resources.
2. Perception runs as a dedicated, daemonized OCR worker process (see
   state_reader.py). The main loop is only the *producer* of frames and the
   *consumer* of parsed state — it never blocks waiting on OCR.
3. The main loop ticks at a minimal cadence (LOOP_INTERVAL), instantly reads the
   latest resources from the shared dict, passes them to the Brain, executes the
   Builder's actions, and repeats.
4. On KeyboardInterrupt the OCR worker is reaped gracefully and the Manager is
   shut down before the process exits.

Safety Mechanisms:
- Global Hotkey: Pressing 'Q' (or configured key) instantly terminates execution.
- The OCR worker is daemonized so it can never outlive the main process.

IMPORTANT (Windows): the entry point is guarded with ``if __name__ == '__main__':``
so that spawned child processes do not re-import and re-execute this module.
"""

import sys
import time
import argparse
import logging
import multiprocessing

# Optional global kill-switch hotkey. This requires admin rights on Windows and is
# therefore imported defensively — the bot still runs without it, relying on
# KeyboardInterrupt (Ctrl+C) as the fallback stop mechanism.
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

from state_reader import GameStateReader
from brain import DecisionMaker
from builder import FortsBuilder

# Configure application-wide logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (Orchestrator): %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fortsbot_runtime.log", mode="w", encoding="utf-8")
    ],
    force=True,  # override any basicConfig that ran during imports (e.g. state_reader)
)

# ==================== BOT SETTINGS ====================
KILL_SWITCH_KEY = "q"        # Hold down this key to immediately stop the bot
LOOP_INTERVAL = 0.005        # Non-blocking decision-loop cadence (seconds)
CAPTURE_INTERVAL = 0.2       # Producer cadence: how often to capture + publish a frame
DECISION_INTERVAL = 1.0      # Consumer cadence: how often to run the brain evaluation
STARTUP_DELAY = 5.0          # Countdown delay to allow window shifting before bot begins


def run_bot(arguments):
    """
    Main orchestrator execution loop (non-blocking producer-consumer).
    """
    logging.info("==================================================")
    print("               FORTSBOT AI INITIATING             ")
    logging.info("==================================================")
    logging.info(f"Non-blocking loop cadence: {LOOP_INTERVAL:.3f}s "
                 f"(capture every {CAPTURE_INTERVAL:.2f}s, decide every {DECISION_INTERVAL:.2f}s)")
    if KEYBOARD_AVAILABLE:
        logging.info(f"Global Safety Kill-Switch Key: '{KILL_SWITCH_KEY.upper()}' (Hold to trigger)")
    else:
        logging.warning("'keyboard' library not found — kill-switch disabled; use Ctrl+C to stop.")

    # 1. Capture Region Definition
    game_region = {
        "top": arguments.window_top,
        "left": arguments.window_left,
        "width": arguments.window_width,
        "height": arguments.window_height
    }
    logging.info(f"Target Game Region Coordinates: {game_region}")

    # 2. Shared state (Producer-Consumer IPC)
    #    A single Manager-backed dict is the source of truth for the decision
    #    layer. The main loop writes the latest parsed resources here and the
    #    Brain reads them back — never blocking on the OCR worker.
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    shared_state["metal"] = 0
    shared_state["energy"] = 0
    shared_state["metal_income"] = 0.0
    shared_state["energy_income"] = 0.0
    shared_state["is_valid"] = False
    shared_state["mine_sites"] = []

    reader = None

    # 3. Module Instantiations
    try:
        logging.info("Initializing Game State Reader (Perception module, daemonized OCR worker)...")
        reader = GameStateReader(screen_region=game_region)

        logging.info("Initializing Builder (Execution module)...")
        builder = FortsBuilder(screen_region=game_region)

        logging.info("Initializing Decision Maker (Brain module)...")
        decision_maker = DecisionMaker(builder=builder)

    except Exception as ex:
        logging.critical(f"Failed to initialize bot modules: {ex}", exc_info=True)
        manager.shutdown()
        sys.exit(1)

    # 4. Preparation Countdown
    logging.info(f"Initialization complete. Starting in {STARTUP_DELAY} seconds...")
    print("\n>>> SWITCH TO YOUR FORTS GAME WINDOW NOW <<<\n")
    for i in range(int(STARTUP_DELAY), 0, -1):
        print(f"Starting in {i}...", end="\r")
        time.sleep(1.0)
    print("BOT IS ACTIVE!                                  ")
    logging.info("Loop active. Commencing non-blocking perception-decision cycles.")

    # 5. Non-blocking execution loop
    last_capture = 0.0
    last_decision = 0.0

    try:
        while True:
            # A. Safety Kill Switch (cheap, non-blocking poll)
            if KEYBOARD_AVAILABLE and keyboard.is_pressed(KILL_SWITCH_KEY):
                logging.warning(f"[TERMINATION] Kill switch '{KILL_SWITCH_KEY.upper()}' detected! Shutting down.")
                break

            now = time.time()

            # A3. Vision step: run YOLO mine-site detection. This is throttled
            #     internally to once per 0.5s so it never pegs the CPU alongside OCR.
            reader.detect_mine_sites()

            # B. Producer step (throttled): capture + publish a fresh grayscale
            #    frame to the OCR worker. The worker drops stale frames (LIFO).
            if now - last_capture >= CAPTURE_INTERVAL:
                reader.capture_and_publish()
                last_capture = now

            # C. Consumer step (non-blocking): read the latest parsed state from
            #    the shared Manager dict — this never waits on OCR.
            state = reader.get_latest_state()

            # D. Decision step (throttled to avoid spamming the brain/execution).
            if now - last_decision >= DECISION_INTERVAL:
                # Publish flat resources into the shared dict for the Brain.
                shared_state["metal"] = state.metal_current
                shared_state["energy"] = state.energy_current
                shared_state["metal_income"] = state.metal_income
                shared_state["energy_income"] = state.energy_income
                shared_state["is_valid"] = state.is_valid

                # Vision results refresh on their own cadence (0.5s) and are always
                # copied here so the Brain never acts on stale coordinates.
                shared_state["mine_sites"] = reader.get_mine_sites()

                # Brain evaluates strategy (build mines when resources + sites align).
                decision_maker.evaluate(shared_state)

                last_decision = now

            # E. Minimal non-blocking sleep keeps the loop tight without pegging CPU.
            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        print("\n[TERMINATION] Interrupted by user (KeyboardInterrupt). Shutting down...")
        logging.info("Bot terminated by user via KeyboardInterrupt.")
    except Exception as general_error:
        logging.critical(f"[CRITICAL ERROR] Bot loop encountered unhandled exception: {general_error}", exc_info=True)
    finally:
        # F. Graceful Shutdown: stop and reap the OCR worker, then the Manager.
        logging.info("==================================================")
        logging.info("             FORTSBOT HAS SHUT DOWN SAFELY        ")
        logging.info("==================================================")
        if reader is not None:
            reader.shutdown()
        manager.shutdown()
        print("[INFO] Bot session concluded. Runtime logs saved to 'fortsbot_runtime.log'.")


def parse_arguments():
    """
    Sets up command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="FortsBot: A behavior-tree-driven macro-strategy automation system for the game 'Forts'."
    )
    parser.add_argument("--window_top", type=int, default=0, help="Y coordinate of top game window edge.")
    parser.add_argument("--window_left", type=int, default=0, help="X coordinate of left game window edge.")
    parser.add_argument("--window_width", type=int, default=3072, help="Width of game window in pixels.")
    parser.add_argument("--window_height", type=int, default=1920, help="Height of game window in pixels.")
    return parser.parse_args()


if __name__ == "__main__":
    # CRITICAL for Windows multiprocessing: process spawning requires the entry
    # point to be guarded, otherwise child processes re-import main.py recursively.
    try:
        args = parse_arguments()
        run_bot(args)
    except PermissionError:
        print("\n[CRITICAL ERROR] Administrative privileges are required.")
        print("                 The 'keyboard' library requires permission to bind a global hotkey on Windows.")
        print("                 Please run your terminal (cmd/powershell) as Administrator and try again:")
        print("                 python main.py")
        sys.exit(1)
