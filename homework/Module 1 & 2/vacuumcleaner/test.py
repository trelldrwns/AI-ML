class VacuumAgent:
    def __init__(self, shape, advantage):
        self.shape = shape
        self.advantage = advantage
        self.is_running = False
        print(f"Initialized a {self.shape}-shaped agent. Advantage: {self.advantage}")

    def start(self):
        self.is_running = True
        print(f"[{self.shape}] Starting cleaning cycle.")

    def stop(self):
        self.is_running = False
        print(f"[{self.shape}] Stopping cleaning cycle.")

    def left(self):
        if self.is_running:
            print(f"[{self.shape}] Turning left.")
        else:
            print(f"[{self.shape}] Agent is stopped. Cannot move.")

    def right(self):
        if self.is_running:
            print(f"[{self.shape}] Turning right.")
        else:
            print(f"[{self.shape}] Agent is stopped. Cannot move.")

    def dock(self):
        self.is_running = False
        print(f"[{self.shape}] Returning to dock to recharge.")

class CircularAgent(VacuumAgent):
    def __init__(self):
        super().__init__("Circular", "Excellent maneuverability, avoids getting stuck.")

class DShapedAgent(VacuumAgent):
    def __init__(self):
        super().__init__("D-Shaped", "Superior edge and corner cleaning.")

class SquareAgent(VacuumAgent):
    def __init__(self):
        super().__init__("Square", "Maximum coverage and deep corner access.")

class TriangularAgent(VacuumAgent):
    def __init__(self):
        super().__init__("Triangular", "Unmatched in tight, non-standard corners.")

d_shaped_vacuum = DShapedAgent()
d_shaped_vacuum.start()
d_shaped_vacuum.left()
d_shaped_vacuum.right()
d_shaped_vacuum.dock()
d_shaped_vacuum.stop()


