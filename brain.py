"""
Forts Strategy Brain (Decision Module)
--------------------------------------
This module implements the decision-making engine of FortsBot. It is built around
a lightweight, object-oriented Behavior Tree (BT) framework.

Architectural Role:
- Acts as the 'Think' layer of the Sense-Think-Act pipeline.
- Consumes the 'GameState' parsed by the State Reader.
- Traverses conditions and composite strategy branches.
- Outputs concrete target construction or repair goals to be processed by the Builder.

Why a Behavior Tree?
Unlike pure state-machines or long 'if/else' chains, Behavior Trees are highly modular,
extensible, and visual. They allow clear prioritizing of tasks (e.g., "always repair
the fort before building weapons").
"""

import enum
import time
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (Brain): %(message)s")


# In-game resource cost of a metal mine (Forts build requirements).
MINE_METAL_COST = 300
MINE_ENERGY_COST = 2000


class NodeStatus(enum.Enum):
    """Execution statuses returned by Behavior Tree nodes."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RUNNING = "RUNNING"


class BlackBoard:
    """
    A shared memory data structure accessible by all nodes in the Behavior Tree.
    Holds references to the GameState, the Builder interface, and any persistent
    cognitive markers (e.g., last built structure, strategic target).
    """
    def __init__(self):
        self.game_state = None   # Current GameState container
        self.builder = None      # Reference to Builder component
        self.last_action = ""    # Description of last ticked action
        self.build_queue = []    # Queued strategic objectives
        self.active_node = None  # Currently running long-running action node (state memory)
        self.mine_sites = []     # Latest YOLO detections: [[x1,y1,x2,y2], ...]


# ==================== BEHAVIOR TREE BASE NODES ====================

class Node:
    """Abstract Base Class for all Behavior Tree nodes."""
    def __init__(self, name="Node"):
        self.name = name

    def tick(self, blackboard):
        """
        Executes node logic. Must be overridden by subclasses.

        Args:
            blackboard (BlackBoard): Shared context reference.

        Returns:
            NodeStatus: Status after execution.
        """
        raise NotImplementedError("Subclasses must implement tick().")


class CompositeNode(Node):
    """Nodes that contain and manage multiple child nodes."""
    def __init__(self, name="Composite", children=None):
        super().__init__(name)
        self.children = children or []

    def add_child(self, child):
        self.children.append(child)


class Selector(CompositeNode):
    """
    A Selector (or Fallback) node runs its children in sequence.
    - If a child succeeds, the Selector succeeds immediately (does not run the rest).
    - If a child returns RUNNING, the Selector returns RUNNING.
    - It fails only if ALL children fail.

    Equivalent to an OR operator / priority-based switch.
    """
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status != NodeStatus.FAILURE:
                return status
        return NodeStatus.FAILURE


class Sequence(CompositeNode):
    """
    A Sequence node runs its children in sequence.
    - If a child fails, the Sequence fails immediately.
    - If a child returns RUNNING, the Sequence returns RUNNING.
    - It succeeds only if ALL children succeed.

    Equivalent to an AND operator.
    """
    def tick(self, blackboard):
        for child in self.children:
            status = child.tick(blackboard)
            if status != NodeStatus.SUCCESS:
                return status
        return NodeStatus.SUCCESS


class Inverter(Node):
    """
    A Decorator node that inverts the result of its single child.
    - SUCCESS ➔ FAILURE
    - FAILURE ➔ SUCCESS
    - RUNNING ➔ RUNNING
    """
    def __init__(self, child, name="Inverter"):
        super().__init__(name)
        self.child = child

    def tick(self, blackboard):
        status = self.child.tick(blackboard)
        if status == NodeStatus.SUCCESS:
            return NodeStatus.FAILURE
        elif status == NodeStatus.FAILURE:
            return NodeStatus.SUCCESS
        return NodeStatus.RUNNING


# ==================== CONCRETE CONDITION NODES ====================

class IsStateValid(Node):
    """Verifies if the perceived GameState is valid (OCR succeeded cleanly)."""
    def tick(self, blackboard):
        if blackboard.game_state and blackboard.game_state.is_valid:
            return NodeStatus.SUCCESS
        logging.warning("Game state is currently invalid. Brain skipped decision tree.")
        return NodeStatus.FAILURE


class IsEnergyStarving(Node):
    """Checks if the fort is in immediate need of power structures."""
    def __init__(self, threshold_percent=0.20):
        super().__init__("IsEnergyStarving")
        self.threshold_percent = threshold_percent

    def tick(self, blackboard):
        state = blackboard.game_state
        # Check if stored energy is below a percentage threshold of max storage,
        # or if the current net income of energy is close to zero/negative.
        current_percent = state.energy_current / max(1, state.energy_max)
        if current_percent < self.threshold_percent or state.energy_income < 10.0:
            logging.info(f"Energy is starving! (Current: {state.energy_current}, Income: +{state.energy_income})")
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class IsMetalSaturated(Node):
    """Checks if we have enough metal to spare for expensive upgrades or weapons."""
    def __init__(self, threshold_value=150):
        super().__init__("IsMetalSaturated")
        self.threshold_value = threshold_value

    def tick(self, blackboard):
        state = blackboard.game_state
        if state.metal_current >= self.threshold_value:
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


class IsUnderAttack(Node):
    """
    Checks if structural components are damaged and need repair.
    In standard game behavior, this checks if damage overlays are present.
    """
    def tick(self, blackboard):
        # Template hook: Real logic can query OpenCV-based damage detection results
        # stored in the game state or blackboard by other perception functions.
        # For this template, let's assume we are under attack if metal income is negative (damage drain).
        if blackboard.game_state.metal_income < -2.0:
            logging.warning("Structure is taking damage (detected negative metal rate)!")
            return NodeStatus.SUCCESS
        return NodeStatus.FAILURE


# ==================== CONCRETE ACTION NODES ====================

class BuildWindTurbine(Node):
    """Instructs the builder module to construct an Energy Wind Turbine."""
    def tick(self, blackboard):
        logging.info("[DECISION] Strategic target met: Ordering Builder to construct Wind Turbine.")

        # In actual implementation, we trigger the builder and pass the target build coordinates
        # and construction menu options:
        if blackboard.builder:
            success = blackboard.builder.build_structure(structure_type="turbine")
            if success:
                blackboard.last_action = "Built Wind Turbine"
                return NodeStatus.SUCCESS
            blackboard.active_node = self
            return NodeStatus.RUNNING

        return NodeStatus.FAILURE


class BuildMetalMine(Node):
    """
    Builds a metal mine at the highest-confidence YOLO-detected build site.

    Requires BOTH: enough metal (>100) AND at least one detected site. A cooldown
    timer (3s by default) prevents re-clicking the same site while the game is
    still constructing. Returns SUCCESS after a click, FAILURE otherwise.
    """
    def __init__(self, cooldown_seconds=3.0):
        super().__init__("BuildMetalMine")
        self.cooldown_seconds = cooldown_seconds
        self.last_build_time = 0.0

    def tick(self, blackboard):
        state = blackboard.game_state
        metal = state.metal_current if state else 0
        energy = state.energy_current if state else 0
        mine_sites = getattr(blackboard, "mine_sites", None) or []

        # Conditions: enough metal, enough energy, and at least one detected site.
        if metal < MINE_METAL_COST or energy < MINE_ENERGY_COST or not mine_sites:
            return NodeStatus.FAILURE

        # Cooldown: ignore calls until it has elapsed.
        now = time.time()
        if now - self.last_build_time < self.cooldown_seconds:
            remaining = self.cooldown_seconds - (now - self.last_build_time)
            logging.info(f"[DECISION] Mine build on cooldown; waiting {remaining:.1f}s.")
            return NodeStatus.FAILURE

        if blackboard.builder:
            x1, y1, x2, y2 = mine_sites[0]  # highest-confidence box first
            logging.info("[DECISION] Strategic target met: Ordering Builder to construct Metal Mine.")
            blackboard.builder.build_mine(x1, y1, x2, y2)
            self.last_build_time = now
            blackboard.last_action = "Built Metal Mine"
            return NodeStatus.SUCCESS

        return NodeStatus.FAILURE


class RepairFortress(Node):
    """Triggers the high-speed repair function on damaged coordinates."""
    def tick(self, blackboard):
        logging.info("[DECISION] Emergency action met: Ordering Builder to run defensive repairs.")

        if blackboard.builder:
            success = blackboard.builder.trigger_repair()
            if success:
                blackboard.last_action = "Executed Repairs"
                return NodeStatus.SUCCESS
            return NodeStatus.RUNNING

        return NodeStatus.FAILURE


class IdleAction(Node):
    """Default fallback state when no strategic conditions are met."""
    def tick(self, blackboard):
        logging.info("[DECISION] Fort is stable. Standing by and accumulating resources...")
        blackboard.last_action = "Idle"
        return NodeStatus.SUCCESS


# ==================== STRATEGY BRAIN ENGINE ====================

class DecisionMaker:
    """
    Evaluates parsed resource state dictionary and makes strategic decisions.
    """
    def __init__(self, builder=None):
        self.builder = builder
        self.last_build_time = 0.0
        self.cooldown_seconds = 3.0   # min seconds between mine placements

    def evaluate(self, shared_state):
        """
        Reads resource values from the shared (Manager-backed) dictionary and makes
        a strategic decision without blocking. Every read uses ``dict.get`` so a
        missing or not-yet-populated key simply yields a safe default — the Brain
        never waits on the perception layer.

        Args:
            shared_state (dict-like): The shared dictionary published by the state
                                      reader via multiprocessing IPC. Expected keys:
                                      "metal", "energy", "metal_income",
                                      "energy_income", "is_valid", "mine_sites".
        """
        if not shared_state:
            print("[WARNING] Shared state dictionary is empty or unavailable.")
            return

        metal = shared_state.get("metal", 0)
        energy = shared_state.get("energy", 0)
        metal_income = shared_state.get("metal_income", 0.0)
        energy_income = shared_state.get("energy_income", 0.0)
        is_valid = shared_state.get("is_valid", False)
        # .get() avoids KeyError if the key has not been populated yet.
        mine_sites = shared_state.get("mine_sites", [])

        # Skip decisions while the OCR worker has not yet produced a clean read.
        if not is_valid:
            return

        print(f"[BRAIN EVAL] Metal: {metal}, Energy: {energy}, "
              f"Metal+{metal_income:.1f}/s, Energy+{energy_income:.1f}/s, "
              f"Sites: {len(mine_sites)}")

        now = time.time()
        if metal >= MINE_METAL_COST and energy >= MINE_ENERGY_COST:
            if not mine_sites:
                print("[ACTION] Resources available but no mine build-site detected yet.")
            elif now - self.last_build_time >= self.cooldown_seconds:
                x1, y1, x2, y2 = mine_sites[0]  # highest-confidence detection first
                if self.builder is not None:
                    self.builder.build_mine(x1, y1, x2, y2)
                else:
                    logging.warning("[DECISION] No builder attached; cannot execute build action.")
                self.last_build_time = now
            else:
                remaining = self.cooldown_seconds - (now - self.last_build_time)
                print(f"[ACTION] Cooldown active, waiting {remaining:.1f} seconds...")
        else:
            print(f"[ACTION] Decision: Waiting for resources "
                  f"(need {MINE_METAL_COST} metal / {MINE_ENERGY_COST} energy)...")


# ==================== LOCAL TEST SCRIPTS ====================
if __name__ == "__main__":
    # Test strategic simulations without active game inputs
    from state_reader import GameState

    print("==================================================")
    print("         FORTS BEHAVIOR TREE BRAIN: TEST          ")
    print("==================================================")

    # Instantiate Brain
    brain = FortsBrain(builder_interface=None)

    # Mock some game scenarios
    print("\n--- SCENARIO 1: Game State is Invalid (OCR Glitch) ---")
    mock_state_invalid = GameState()
    mock_state_invalid.is_valid = False
    brain.tick(mock_state_invalid)

    print("\n--- SCENARIO 2: Fort Under Attack (Negative Metal Income) ---")
    mock_state_attack = GameState()
    mock_state_attack.is_valid = True
    mock_state_attack.metal_current = 200
    mock_state_attack.metal_income = -5.0
    brain.tick(mock_state_attack)

    print("\n--- SCENARIO 3: Energy Starvation ---")
    mock_state_energy = GameState()
    mock_state_energy.is_valid = True
    mock_state_energy.energy_current = 1000  # 10% of 9000 limit
    mock_state_energy.energy_income = 4.0
    mock_state_energy.metal_current = 50
    mock_state_energy.metal_income = 2.0
    brain.tick(mock_state_energy)

    print("\n--- SCENARIO 4: High Metal Excess ---")
    mock_state_excess = GameState()
    mock_state_excess.is_valid = True
    mock_state_excess.energy_current = 5000
    mock_state_excess.energy_income = 30.0
    mock_state_excess.metal_current = 180  # exceeds threshold
    mock_state_excess.metal_income = 5.0
    brain.tick(mock_state_excess)
