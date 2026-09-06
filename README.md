# FortsBot: Behavior-Tree Macro-Strategy AI

An open-source, modular, hardcoded macro-strategy bot for the 2D physics-based game **Forts**. 

Unlike machine learning models that require massive datasets and lack transparency, **FortsBot** is built as an **Expert System** driven by a **Behavior Tree (BT)**. This architecture ensures high determinism, real-time debugging, and precise strategic control, allowing developers to craft and fine-tune complex build-orders, defense protocols, and aiming algorithms.

---

## 👁️ System Architecture

FortsBot follows a classic robotic paradigm: **Sense ➔ Think ➔ Act**.

```
    +--------------------------------------------------+
    |                    PC SCREEN                     |
    +--------------------------------------------------+
                             |  (Screen Capture via mss)
                             v
    +--------------------------------------------------+
    |            PERCEPTION (state_reader)             |
    |  - ROIs for Metal & Energy readouts              |
    |  - Image preprocessing (Gray, Scale, Threshold)  |
    |  - OCR Engine (Tesseract/EasyOCR)                |
    +--------------------------------------------------+
                             |  (Current Game State)
                             v
    +--------------------------------------------------+
    |               DECISION (brain)                   |
    |  - Custom, lightweight Behavior Tree (BT)        |
    |  - Composite Nodes (Sequence, Selector)          |
    |  - Conditions (Metal/Energy checks, etc.)        |
    |  - Strategy Actions (Build Mine, Upgrade, etc.)  |
    +--------------------------------------------------+
                             |  (Action Commands)
                             v
    +--------------------------------------------------+
    |              EXECUTION (builder)                 |
    |  - Precise PyAutoGUI & PyDirectInput simulation   |
    |  - Structure building (node coordinates, drags)  |
    |  - Safe coordinate scaling and global kill switch |
    +--------------------------------------------------+
```

### 1. Perception (`state_reader.py`)
Uses high-performance screen capture to extract regions of interest (ROIs) from the game HUD. It processes these images using **OpenCV** (grayscale, thresholding, and scale-ups) and feeds them to an OCR engine (**PyTesseract** or **EasyOCR**) to track:
*   **Metal Storage & Income** (e.g., `125 / 1000 (+4.2)`)
*   **Energy Storage & Income** (e.g., `2400 / 9000 (+35)`)
*   **Game Time & Phase** (detecting build-phase vs. combat-phase)

### 2. Decision Making (`brain.py`)
Utilizes an elegant object-oriented **Behavior Tree** representing the macro-strategy:
*   **Selectors (Fallback)**: Evaluate children until one succeeds (e.g., "Check emergency defenses first; if no emergency, execute normal build order").
*   **Sequences**: Evaluate children in order until one fails (e.g., "Verify metal is high AND energy is high AND space is available THEN build weapon upgrade").
*   **Action & Condition Nodes**: Interface directly with the perceived state and translate choices into execution goals.

### 3. Execution (`builder.py`)
Translates high-level decisions (e.g., `BuildWindTurbine`) into direct keyboard and mouse controls:
*   **DirectInput Simulation**: Utilizing `pydirectinput` for compatibility with DirectX-based rendering.
*   **Spatial Mapping**: Controls building on structural nodes (clicking, drawing structural beams, placing weapons, and aligning upgrade vectors).
*   **Failsafe Mechanics**: Automatically pauses or stops if the human operator moves the mouse to the corner of the screen or holds down the emergency halt key (`Q`).

---

## 📂 Project Directory Structure

```text
FortsBot/
│
├── main.py            # Bot Entry Point (Initializes modules & runs the game loop)
├── state_reader.py    # Screen Capture, ROI Extraction, and OCR Preprocessing
├── brain.py           # Behavior Tree framework & Strategic Logic Nodes
├── builder.py         # Hardware Mouse/Keyboard Simulation and DirectInput clicks
│
├── forts_repair_bot.py# Independent legacy Repair Bot utility
└── README.md          # Open-source Project Overview and Documentation
```

---

## 🛠️ Setup & Installation

### Prerequisites
1.  **Python 3.8+** (64-bit recommended)
2.  **Tesseract OCR** (Required if using PyTesseract)
    *   **Windows**: Download the installer from UB Mannheim. Add the installation folder (e.g., `C:\Program Files\Tesseract-OCR`) to your system environment variables (`PATH`), or point to it directly in the configuration.
    *   **Linux**: `sudo apt install tesseract-ocr`

### Installation
Clone the repository and install dependencies within your virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-username/FortsBot.git
cd FortsBot

# Activate your virtual environment (if using .venv)
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install required Python packages
pip install mss opencv-python numpy pytesseract easyocr pyautogui pydirectinput keyboard
```

---

## 🚀 How to Run

1.  Start your **Forts** game in Windowed or Borderless Fullscreen mode.
2.  Launch the bot entry point from an Administrator terminal (needed for `keyboard` and `pydirectinput` to simulate hardware signals):
    ```bash
    python main.py
    ```
3.  **To Stop the Bot**: Hold down the **`Q`** key or slide your mouse cursor quickly to any corner of the screen (triggering PyAutoGUI's Failsafe).

---

## 📝 Roadmap & Contribution

We are building a highly capable, modular, and fair-play offline/sandbox practice bot. Contributions are highly welcome!
- [ ] Add coordinate calibration tools for different screen resolutions (1080p, 1440p).
- [ ] Build a visual debugger overlay showing the active branch of the Behavior Tree.
- [ ] Create pre-configured macro templates (e.g., "Laser Rush", "Howitzer Spam", "Turtle Defense").
- [ ] Implement advanced structural building (physics validation, beam angles, armor layer drawing).

---

## ⚖️ License & Fair Play

This project is licensed under the MIT License. This software is built strictly for **educational purposes, offline sandbox testing, and AI research**. Do not use this tool in online matchmaking, as it violates the multiplayer terms of service of most platforms and games.
