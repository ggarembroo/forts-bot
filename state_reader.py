"""
Forts Game State Reader (Perception Module)
-------------------------------------------
This module captures the game screen and extracts structured game state data
(such as Metal and Energy quantities, income, and time) using Computer Vision
and Optical Character Recognition (OCR).

Architectural Role:
- Acts as the 'Sense' layer of the Sense-Think-Act pipeline.
- Performs high-speed cropping of standard game HUD regions of interest (ROIs).
- Applies image preprocessing (scaling, binarization, morphological filtering)
  specifically tuned for OCR text isolation.
- Uses lightweight Tesseract OCR (via pytesseract) for fast <50ms per frame reading.

Concurrency Model (Producer-Consumer, multiprocessing):
---------------------------------------------------------
OCR is isolated into a dedicated ``multiprocessing.Process`` (NOT a thread) so the
GIL can never stall the main game loop. The producer (main process) captures the
screen with ``mss``, converts the frame to grayscale with ``cv2`` to minimise the
IPC payload, and publishes it into a ``multiprocessing.Manager().dict()``.

LIFO / frame-dropping: the shared dict holds exactly ONE "latest" frame keyed by a
monotonic ``frame_id``. A slow OCR worker simply skips every intermediate frame and
always processes the absolute newest frame — stale frames are overwritten, never
queued, which prevents state desync between perception and decision.

The parsed ``GameState`` is written back into the same shared dict, from which the
main loop reads it with ``get_nowait()``-style zero-blocking semantics.
"""

import threading
import time
import logging
import re
import multiprocessing

import cv2
import numpy as np
import mss

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (StateReader): %(message)s")

# Import Tesseract OCR
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

# Tesseract is configured to read a single line of digits only. This dramatically
# reduces false positives on HUD textures (colons, slashes, letters are ignored).
OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"

# --- YOLO vision (mine build-site detection) ---
YOLO_MODEL_PATH = "yolo_test/best.pt"   # Trained YOLOv8n (class: mine_build_site, nc=1)
YOLO_CONF = 0.4                             # Minimum confidence for a valid detection
YOLO_INTERVAL = 0.5                         # Run inference at most every 0.5s (CPU budget)


class GameState:
    """
    A simple container representing the parsed variables of the game.
    Passed downstream to the Behavior Tree (brain) for strategic decisions.
    """
    def __init__(self):
        self.metal_current = 0
        self.metal_max = 1000
        self.metal_income = 0.0

        self.energy_current = 0
        self.energy_max = 9000
        self.energy_income = 0.0

        self.game_time_seconds = 0.0
        self.is_valid = False  # Set to True if reading succeeded without anomalies

    def __repr__(self):
        return (f"<GameState Metal: {self.metal_current}/{self.metal_max} (+{self.metal_income:0.1f}) | "
                f"Energy: {self.energy_current}/{self.energy_max} (+{self.energy_income:0.1f}) | "
                f"Time: {self.game_time_seconds:.1f}s | Valid: {self.is_valid}>")


def preprocess_roi(gray_roi):
    """
    Applies image processing optimizations specifically to make screen text
    highly readable for OCR engines (especially numbers in game interfaces).

    NOTE: Input is expected to already be a single-channel grayscale ROI (the full
    frame is converted to grayscale once at capture time, before cropping).

    Args:
        gray_roi (np.ndarray): Grayscale cropped image (2D uint8).

    Returns:
        np.ndarray: Binarized, cleaned grayscale image.
    """
    # 1. Resize/Up-scale to increase OCR resolution (2.5x works best for HUD numbers)
    scaled = cv2.resize(gray_roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # 2. Apply Gaussian Blur to smooth transition edges
    blurred = cv2.GaussianBlur(scaled, (3, 3), 0)

    # 3. Apply Otsu's Thresholding
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Invert image so text is black on a white background (Tesseract preference)
    thresholded = cv2.bitwise_not(thresholded)

    # 5. Morphological cleaning to connect split numbers
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    return cleaned


def _process_frame(gray, rois):
    """
    Runs ROI cropping (numpy) + preprocessing + OCR + regex parsing on a single
    grayscale frame and returns a fully parsed ``GameState``. This function runs
    exclusively inside the dedicated OCR worker process.
    """
    state = GameState()
    success_count = 0
    frame_h, frame_w = gray.shape[:2]

    for name, roi in rois.items():
        rx, ry, rw, rh = roi["x"], roi["y"], roi["w"], roi["h"]
        if rx + rw > frame_w or ry + rh > frame_h:
            logging.warning(f"ROI '{name}' bounds exceed captured frame dimensions.")
            continue

        # ROI cropping is done with numpy slicing on the shared grayscale frame.
        cropped = gray[ry:ry + rh, rx:rx + rw]
        preprocessed = preprocess_roi(cropped)
        raw_text = pytesseract.image_to_string(preprocessed, config=OCR_CONFIG).strip()

        # ---- Parse each HUD field (logic preserved from the original reader) ----
        if name == "metal_value":
            match = re.search(r'\d+', raw_text)
            if match:
                state.metal_current = int(match.group())
                state.metal_max = max(1000, state.metal_current)
                success_count += 1

        elif name == "metal_income":
            match = re.search(r'\d+', raw_text)
            if match:
                state.metal_income = float(match.group())

        elif name == "energy_value":
            match = re.search(r'\d+', raw_text)
            if match:
                state.energy_current = int(match.group())
                state.energy_max = max(9000, state.energy_current)
                success_count += 1

        elif name == "energy_income":
            match = re.search(r'\d+', raw_text)
            if match:
                state.energy_income = float(match.group())

        elif name == "game_time":
            raw_time = raw_text
            clean_time = "".join(char for char in raw_time if char.isdigit() or char in ".:")
            try:
                if ":" in clean_time:
                    minutes, seconds = clean_time.split(":")[:2]
                    state.game_time_seconds = int(minutes) * 60 + float(seconds)
                elif clean_time:
                    state.game_time_seconds = float(clean_time)
                success_count += 1
            except ValueError:
                state.game_time_seconds = 0.0

    if success_count >= 2:
        state.is_valid = True

    return state


def _ocr_worker(shared, rois):
    """
    Dedicated OCR worker process.

    Continuously polls the shared Manager dict for a NEW grayscale frame. Because
    the producer overwrites the same dict keys each capture, this worker never
    accumulates a backlog: it compares the current ``frame_id`` against the last
    frame it processed and, if it has fallen behind, silently drops all stale
    frames and processes only the absolute latest (LIFO / overwrite semantics).

    The parsed ``GameState`` is written back into ``shared["state"]``.

    Args:
        shared (DictProxy): Shared Manager dict used for IPC.
        rois (dict): ROI coordinate definitions to crop from each frame.
    """
    if not PYTESSERACT_AVAILABLE:
        logging.error("OCR worker started but pytesseract/Tesseract CLI is unavailable.")
        shared["stop"] = True
        return

    last_processed_id = -1
    logging.info(f"[OCR Worker] Started (pid={multiprocessing.current_process().pid}). "
                 f"Processing ROIs: {list(rois)}")

    try:
        while not shared["stop"]:
            # Cheap read first: only fetch the (potentially large) frame when the
            # producer has published a newer frame_id than the one we just processed.
            frame_id = shared["frame_id"]
            if frame_id <= last_processed_id:
                time.sleep(0.003)  # No new work; yield before polling again.
                continue

            gray = shared["frame"]
            if gray is None:
                time.sleep(0.003)
                continue

            # Mark this frame as consumed BEFORE the (slow) OCR pass. Any frames the
            # producer publishes while we work will simply advance frame_id past us
            # and be processed on the next iteration — the interim ones are dropped.
            last_processed_id = frame_id

            try:
                state = _process_frame(gray, rois)
            except Exception as exc:
                logging.error(f"[OCR Worker] Frame processing failed: {exc}", exc_info=True)
                state = GameState()

            # Publish the parsed state back to the main process (single atomic write).
            shared["state"] = state
    except KeyboardInterrupt:
        # On Windows, Ctrl+C is broadcast to the whole console process group, so
        # this worker receives it too (usually while blocked inside a Tesseract
        # subprocess call). The parent is already shutting down — fall through to
        # a clean exit instead of dumping a traceback.
        pass

    logging.info("[OCR Worker] Stop flag received. Shutting down.")


class GameStateReader:
    """
    Captures the screen in the main process and farms OCR out to a dedicated
    ``multiprocessing.Process``. The main loop only ever publishes the latest
    grayscale frame and reads the latest parsed state — never blocking on OCR.
    """
    def __init__(self, screen_region=None):
        """
        Initializes the game state reader.

        Args:
            screen_region (dict): The target window region to capture. Format:
                                  {"top": int, "left": int, "width": int, "height": int}
                                  If None, defaults to 3072x1920 display resolution.
        """
        # Hardcode the screen region to 3072x1920 resolution if no capture region is supplied
        self.screen_region = screen_region or {"top": 0, "left": 0, "width": 3072, "height": 1920}
        self.sct = mss.MSS()

        # Validate Tesseract availability
        if not PYTESSERACT_AVAILABLE:
            logging.error("pytesseract package or system Tesseract CLI not found! Please install pytesseract and Tesseract-OCR.")
            raise ImportError("Ensure pytesseract is installed to run StateReader.")
        logging.info("Tesseract OCR engine selected and validated.")

        # Calculate ROIs dynamically using relative percentages based on screen width and height (3072x1920 box)
        w = self.screen_region["width"]
        h = self.screen_region["height"]

        self.rois = {
            "metal_value": {"x": 2664, "y": 1668, "w": 168, "h": 75},
            "metal_income": {"x": 2952, "y": 1692, "w": 72, "h": 49},
            "energy_value": {"x": 2662, "y": 1762, "w": 170, "h": 76},
            "energy_income": {"x": 2954, "y": 1787, "w": 69, "h": 49},
            "game_time": {"x": 1449, "y": 19, "w": 199, "h": 59}
        }

        # ---- IPC setup: a shared Manager dict replaces both a frame queue and a
        # result queue. It holds exactly one "latest" frame, so LIFO/overwrite is
        # implicit — the producer can never outpace the consumer into a backlog.
        self.manager = multiprocessing.Manager()
        self.shared = self.manager.dict()
        self.shared["stop"] = False          # Graceful-shutdown flag for the worker
        self.shared["frame_id"] = -1         # Monotonic id of the latest frame
        self.shared["frame"] = None          # Latest grayscale frame (numpy array)
        self.shared["state"] = None          # Latest parsed GameState
        self.shared["mine_sites"] = []       # Latest YOLO detections: [[x1,y1,x2,y2], ...]

        # Spawn the dedicated OCR process (Windows-safe: spawn, not fork).
        self.process = multiprocessing.Process(
            target=_ocr_worker,
            args=(self.shared, self.rois),
            daemon=True,
            name="ocr-worker"
        )
        self.process.start()

        # --- YOLO vision model ---
        # Loaded lazily here (NOT at module import) so the spawned OCR worker
        # process never pays the ultralytics import cost. The model lives in the
        # main process and is driven by detect_mine_sites().
        self.model = None
        self._last_yolo_time = 0.0          # throttle timer for YOLO_INTERVAL
        self._mine_sites_lock = threading.Lock()
        try:
            from ultralytics import YOLO
            self.model = YOLO(YOLO_MODEL_PATH)
            logging.info(f"YOLO vision model loaded from '{YOLO_MODEL_PATH}'")
        except Exception as exc:
            logging.error(f"Failed to load YOLO model '{YOLO_MODEL_PATH}': {exc} "
                          f"(vision disabled; continuing in OCR-only mode)")

        logging.info(f"GameStateReader initialized with screen region: {self.screen_region}")
        logging.info(f"OCR worker process started (pid={self.process.pid}). "
                     f"ROIs for resolution {w}x{h}: {self.rois}")

    def _grab_color_frame(self):
        """
        Captures the defined screen region with mss and returns the ORIGINAL color
        frame in BGR (the raw screenshot, before any grayscale conversion).
        """
        screenshot = self.sct.grab(self.screen_region)
        frame = np.array(screenshot)
        # mss returns BGRA; convert to standard BGR for OpenCV.
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def capture_frame(self):
        """
        Captures the defined screen region and immediately converts it to grayscale
        to minimise the IPC payload sent to the OCR worker.

        Returns:
            np.ndarray: Single-channel grayscale OpenCV frame.
        """
        bgr = self._grab_color_frame()
        # Grayscale is the smallest payload that still contains everything OCR needs
        # (luminance of the HUD digits).
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def publish_frame(self, gray):
        """
        Publishes the latest grayscale frame to the shared dict with LIFO semantics.
        Ordering matters: write the (large) frame first, then the (small) frame_id,
        so the worker never observes a new id before the matching frame is committed.
        """
        self.shared["frame"] = gray
        self.shared["frame_id"] += 1

    def capture_and_publish(self):
        """
        Producer step: capture one frame, convert it to grayscale, and publish it
        to the OCR worker. The previous frame is overwritten (LIFO semantics), so
        a slow OCR worker always ends up processing the newest frame only.
        """
        bgr = self._grab_color_frame()
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.publish_frame(gray)

    def detect_mine_sites(self):
        """
        Runs the YOLO vision model on a fresh color frame to locate mine build-sites.

        Throttled to YOLO_INTERVAL (default 0.5s) so the CPU is never pegged; OCR
        keeps its own independent cadence. The result is written into
        shared["mine_sites"] as a list of [x1, y1, x2, y2] absolute screen-pixel
        boxes, sorted by descending confidence (best detection first). If nothing
        is found (or inference fails), an empty list is written so the Brain never
        acts on stale coordinates.
        """
        if self.model is None:
            return

        now = time.time()
        if now - self._last_yolo_time < YOLO_INTERVAL:
            return
        self._last_yolo_time = now

        bgr = self._grab_color_frame()
        try:
            results = self.model(bgr, conf=YOLO_CONF, verbose=False)[0]
        except Exception as exc:
            logging.error(f"[Vision] YOLO inference failed: {exc}")
            with self._mine_sites_lock:
                self.shared["mine_sites"] = []
            return

        offset_x = self.screen_region["left"]
        offset_y = self.screen_region["top"]
        region_w = self.screen_region["width"]
        region_h = self.screen_region["height"]

        sites = []
        for box in results.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])

            # A real box must be non-degenerate.
            if x1 >= x2 or y1 >= y2:
                continue

            # YOLO returns coordinates relative to the captured frame; offset them
            # into absolute screen pixels for the Builder's click layer.
            sx1 = int(x1) + offset_x
            sy1 = int(y1) + offset_y
            sx2 = int(x2) + offset_x
            sy2 = int(y2) + offset_y

            # Validate: keep the box inside the captured screen region.
            if (sx1 < offset_x or sy1 < offset_y
                    or sx2 > offset_x + region_w or sy2 > offset_y + region_h):
                continue

            sites.append((conf, [sx1, sy1, sx2, sy2]))

        # Best (highest-confidence) detection first.
        sites.sort(key=lambda item: item[0], reverse=True)

        logging.info(f"[Vision] YOLO видит объектов: {len(results.boxes)} "
                     f"(conf >= {YOLO_CONF}); валидных рамок шахт: {len(sites)}")

        with self._mine_sites_lock:
            self.shared["mine_sites"] = [box for _, box in sites]

    def get_mine_sites(self):
        """
        Returns the latest detected mine build-sites as a list of
        [x1, y1, x2, y2] screen-pixel boxes, or an empty list if none were found.
        """
        with self._mine_sites_lock:
            return list(self.shared.get("mine_sites", []))

    def get_latest_state(self):
        """
        Consumer step (100% non-blocking): return the most recently parsed
        GameState from the OCR worker without capturing or waiting on OCR.
        Returns an invalid GameState if the worker has not produced a result yet.
        """
        state = self.shared["state"]
        if state is None:
            return GameState()
        return state

    def read_game_state(self):
        """
        Convenience wrapper combining the producer and consumer steps in one call.
        Kept for backwards compatibility with the synchronous test harness.
        """
        self.capture_and_publish()
        return self.get_latest_state()

    def shutdown(self):
        """
        Gracefully terminates the OCR worker process and the Manager.

        Order of operations:
        1. Set the stop flag so the worker exits its loop cleanly.
        2. Join the worker (give it a short grace period).
        3. Force ``terminate()`` if it is still stuck inside a Tesseract call.
        4. Shut down the Manager to release the shared-memory resources.
        """
        logging.info("Shutting down OCR worker and shared memory...")
        self.shared["stop"] = True

        if self.process.is_alive():
            self.process.join(timeout=2.0)

        if self.process.is_alive():
            logging.warning("OCR worker did not exit gracefully; force-terminating.")
            self.process.terminate()
            self.process.join(timeout=1.0)

        self.manager.shutdown()
        logging.info("StateReader shutdown complete.")


# ==================== LOCAL TEST SCRIPTS ====================
if __name__ == "__main__":
    print("==================================================")
    print("      FORTS STATE READER (TESSERACT): TEST RUN    ")
    print("==================================================")
    print("System will try to read from top-left screen region.")
    print("Launch 'Forts' game or open a matching mockup screen!")
    print("Press Ctrl+C to terminate.")
    print("--------------------------------------------------")

    try:
        print(f"MSS Monitors: {mss.MSS().monitors}")
        reader = GameStateReader()

        print("Switching to game window in 3 seconds...")
        time.sleep(3)

        while True:
            start_time = time.time()
            state = reader.read_game_state()
            elapsed = time.time() - start_time

            print(f"[READ STATE] {state} | Loop Latency: {elapsed*1000:.1f}ms")
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[INFO] Test run completed by user.")
    except Exception as ex:
        print(f"\n[CRITICAL] Error starting GameStateReader: {ex}")
    finally:
        try:
            reader.shutdown()
        except Exception:
            pass
