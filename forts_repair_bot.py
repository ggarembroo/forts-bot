"""
Forts Auto-Repair Bot
---------------------
An automated computer vision bot for the physics-based game "Forts".
It captures the game screen, detects damaged structure points using HSV color masking,
and simulates high-speed mouse clicks to repair them.

Features:
- High-speed screen capture via 'mss'.
- Fast color and contour detection via 'OpenCV'.
- Click simulation via 'pyautogui' and 'pydirectinput' (for DirectX games).
- Interactive HSV Tuner Mode with sliders to easily find the perfect color values.
- Global Kill-Switch hotkey to immediately halt execution.

Tech Stack:
- mss: High-speed screen capture.
- opencv-python (cv2): Color space conversion, morphological cleaning, and contour detection.
- numpy: Array thresholds.
- pyautogui / pydirectinput: Input actions.
- keyboard: Global keyboard hook for safety shutdown.

Author: Expert Python Developer (AI Assistant)
Date: 2026-08-30
"""

import time
import sys
import numpy as np
import cv2
import mss
import keyboard
import pyautogui

# Adjust pyautogui parameters for high-speed responsiveness and safety
pyautogui.PAUSE = 0.02  # Minimal delay between actions
pyautogui.FAILSAFE = True  # Move mouse to any screen corner to trigger FailSafeException

# For modern games built on DirectX or OpenGL, standard pyautogui clicks sometimes fail.
# pydirectinput uses DirectInput API to properly simulate physical keyboard/mouse signals.
try:
    import pydirectinput
    pydirectinput.PAUSE = 0.02
    USE_DIRECTINPUT = True
    print("[INFO] pydirectinput detected and will be used for clicks.")
except ImportError:
    USE_DIRECTINPUT = False
    print("[WARNING] pydirectinput not found. Falling back to pyautogui.")
    print("          If clicks do not register in-game, please run: pip install pydirectinput")


# ==================== CONFIGURATION ====================

# 1. SCREEN REGION TO CAPTURE
# Adjust this to match your game window bounds.
# Examples:
# - Full Screen (1920x1080): {"top": 0, "left": 0, "width": 1920, "height": 1080}
# - Windowed mode: specify the exact coordinates of the game client.
CAPTURE_REGION = {
    "top": 0,
    "left": 0,
    "width": 1920,
    "height": 1080
}

# 2. VISION SETTINGS (HSV COLOR RANGE FOR DAMAGED WOOD)
# OpenCV HSV range: H [0-179], S [0-255], V [0-255].
# The colors below are placeholder thresholds representing wood/orange-brown.
#
# HOW TO TUNE:
# - Set MODE = "tuner" below.
# - Run the script. An interactive window with sliders will appear.
# - Adjust sliders until ONLY the damaged targets appear white on the mask.
# - Once isolated, copy the printed values back into LOWER_HSV and UPPER_HSV below.
LOWER_HSV = np.array([10, 100, 100])   # Default lower bound (orange-ish/brown)
UPPER_HSV = np.array([25, 255, 255])   # Default upper bound (orange-ish/brown)

# 3. CONTOUR FILTERING
# Filters out tiny noise particles or huge UI panels.
MIN_CONTOUR_AREA = 100   # Minimum pixel area of a damaged spot
MAX_CONTOUR_AREA = 4000  # Maximum pixel area of a damaged spot

# 4. BOT OPERATION SETTINGS
MODE = "bot"             # Options: "bot" (runs the clicker) OR "tuner" (interactive HSV slider mode)
CLICK_COOLDOWN = 0.15    # Delay (seconds) between clicks to prevent over-clicking
DEBUG_VIEW = True        # If True, displays visual windows of what the bot sees (slower but good for debugging)
KILL_SWITCH_KEY = "q"    # Global hotkey to instantly shut down the script (hold down to trigger)

# =======================================================


def click_target(x_screen, y_screen):
    """
    Performs a hardware-level click at the absolute screen coordinates.
    """
    print(f"[ACTION] Clicking damaged structure at: ({x_screen}, {y_screen})")
    if USE_DIRECTINPUT:
        pydirectinput.click(x_screen, y_screen)
    else:
        # Fallback to standard pyautogui
        pyautogui.click(x_screen, y_screen)


def process_frame(frame):
    """
    Processes the raw capture frame:
    - Converts it to HSV.
    - Masks the target HSV range.
    - Cleans noise with morphological opening/closing.
    - Finds contour centers that fall within the size limits.
    """
    # 1. Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. Filter target colors
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    # 3. Perform morphology to reduce noise specs and fill gaps
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4. Detect contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    targets = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
            # Calculate contour centroid (center) using spatial moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                targets.append((cX, cY))

                # Draw circles on debug view
                if DEBUG_VIEW:
                    cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
                    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"Damaged: {int(area)}px", (cX - 30, cY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return targets, mask


def run_bot():
    """
    Main active loop of the auto-repair clicker bot.
    """
    print("==================================================")
    print("           FORTS AUTO-REPAIR BOT: RUNNING         ")
    print("==================================================")
    print(f"[-] Global Kill Switch: Hold down '{KILL_SWITCH_KEY.upper()}' to STOP and EXIT.")
    print(f"[-] Debug Screen: {'ENABLED' if DEBUG_VIEW else 'DISABLED'}")
    print(f"[-] Capture Window Bounds: {CAPTURE_REGION}")
    print("--------------------------------------------------")
    print("Starting in 3 seconds... Switch to your game window!")
    time.sleep(3)
    print("[RUNNING] Bot active. Scanning and repairing...")

    last_click_time = 0

    with mss.mss() as sct:
        while True:
            # 1. Check Safety Kill Switch first
            if keyboard.is_pressed(KILL_SWITCH_KEY):
                print(f"\n[TERMINATED] Kill switch '{KILL_SWITCH_KEY.upper()}' pressed. Stopping bot...")
                break

            # 2. Capture defined screen area
            screenshot = sct.grab(CAPTURE_REGION)

            # Convert mss screenshot object to numpy/OpenCV array format
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # 3. Process frame to discover targets
            targets, mask = process_frame(frame)

            # 4. Click target if cooldown has expired
            current_time = time.time()
            if targets and (current_time - last_click_time >= CLICK_COOLDOWN):
                # Target first detected spot (coordinates relative to captured region)
                t_x, t_y = targets[0]

                # Transform to absolute screen coordinates
                screen_x = CAPTURE_REGION["left"] + t_x
                screen_y = CAPTURE_REGION["top"] + t_y

                # Fire the click action
                click_target(screen_x, screen_y)
                last_click_time = current_time

            # 5. Display visual feeds if Debug View is enabled
            if DEBUG_VIEW:
                # Resize previews to fit comfortably on screen
                preview_frame = cv2.resize(frame, (800, 450))
                preview_mask = cv2.resize(mask, (800, 450))

                cv2.imshow("Bot View - Game Capture", preview_frame)
                cv2.imshow("Bot View - Filtered Mask", preview_mask)

                # Required for OpenCV window drawings to register.
                # Pressing ESC inside an OpenCV window will also exit.
                if cv2.waitKey(1) & 0xFF == 27:
                    print("\n[TERMINATED] Debug window closed. Stopping bot...")
                    break

    cv2.destroyAllWindows()
    print("[FINISHED] Bot has shut down safely.")


def run_tuner():
    """
    Opens an interactive GUI with sliders to calibrate LOWER_HSV and UPPER_HSV.
    """
    print("\n==================================================")
    print("              INTERACTIVE HSV TUNER               ")
    print("==================================================")
    print("[-] Drag the sliders until ONLY the targets are WHITE.")
    print(f"[-] Hold down '{KILL_SWITCH_KEY.upper()}' or press ESC on the window to save and exit.")
    print("--------------------------------------------------")

    def nothing(x):
        pass

    # Create slider trackbar GUI window
    cv2.namedWindow("HSV Control Sliders")
    cv2.resizeWindow("HSV Control Sliders", 400, 320)

    # Initialize sliders to current configurations
    cv2.createTrackbar("Min Hue", "HSV Control Sliders", LOWER_HSV[0], 179, nothing)
    cv2.createTrackbar("Max Hue", "HSV Control Sliders", UPPER_HSV[0], 179, nothing)
    cv2.createTrackbar("Min Saturation", "HSV Control Sliders", LOWER_HSV[1], 255, nothing)
    cv2.createTrackbar("Max Saturation", "HSV Control Sliders", UPPER_HSV[1], 255, nothing)
    cv2.createTrackbar("Min Value", "HSV Control Sliders", LOWER_HSV[2], 255, nothing)
    cv2.createTrackbar("Max Value", "HSV Control Sliders", UPPER_HSV[2], 255, nothing)

    # Initialize coordinates variables for printing results at end
    min_h, max_h = LOWER_HSV[0], UPPER_HSV[0]
    min_s, max_s = LOWER_HSV[1], UPPER_HSV[1]
    min_v, max_v = LOWER_HSV[2], UPPER_HSV[2]

    with mss.mss() as sct:
        while True:
            if keyboard.is_pressed(KILL_SWITCH_KEY):
                break

            screenshot = sct.grab(CAPTURE_REGION)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Get slider value positions
            min_h = cv2.getTrackbarPos("Min Hue", "HSV Control Sliders")
            max_h = cv2.getTrackbarPos("Max Hue", "HSV Control Sliders")
            min_s = cv2.getTrackbarPos("Min Saturation", "HSV Control Sliders")
            max_s = cv2.getTrackbarPos("Max Saturation", "HSV Control Sliders")
            min_v = cv2.getTrackbarPos("Min Value", "HSV Control Sliders")
            max_v = cv2.getTrackbarPos("Max Value", "HSV Control Sliders")

            lower = np.array([min_h, min_s, min_v])
            upper = np.array([max_h, max_s, max_v])

            # Apply thresholds
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            tuner_mask = cv2.inRange(hsv_img, lower, upper)

            # Morphological clean
            kernel = np.ones((3, 3), np.uint8)
            tuner_mask = cv2.morphologyEx(tuner_mask, cv2.MORPH_OPEN, kernel)
            tuner_mask = cv2.morphologyEx(tuner_mask, cv2.MORPH_CLOSE, kernel)

            # Convert 1-channel mask to 3-channel to merge side-by-side with game feed
            mask_3ch = cv2.cvtColor(tuner_mask, cv2.COLOR_GRAY2BGR)
            combined_display = np.hstack([frame, mask_3ch])

            # Resize to fit normal monitors
            display_resized = cv2.resize(combined_display, (1200, 450))

            cv2.imshow("HSV Tuner (LEFT: Game capture, RIGHT: Mask filter)", display_resized)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()

    print("\n==================================================")
    print("               CALIBRATION RESULTS                 ")
    print("==================================================")
    print("Copy and paste these values into the CONFIGURATION section at the top of the script:")
    print(f"LOWER_HSV = np.array([{min_h}, {min_s}, {min_v}])")
    print(f"UPPER_HSV = np.array([{max_h}, {max_s}, {max_v}])")
    print("==================================================")


if __name__ == "__main__":
    # Ensure Windows console users have high visibility on administrator rules for keyboard hooking
    try:
        if MODE.strip().lower() == "tuner":
            run_tuner()
        else:
            run_bot()
    except PermissionError:
        print("\n[ERROR] Keyboard library needs administrative privileges to capture global hotkeys on Windows.")
        print("        Please open your Command Prompt / PowerShell as Administrator and run the script again:")
        print("        python forts_repair_bot.py")
