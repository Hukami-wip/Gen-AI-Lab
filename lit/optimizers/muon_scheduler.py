import math
from typing import Optional


class MuonScheduler:
    """
    Per-group scheduler for SingleDeviceMuonWithAuxAdam.

    Features:
      - Muon cosine LR with warmup and floor
      - Optional scalar cosine (or constant)
      - Momentum annealing for Muon groups (piecewise or cosine)
      - Supports resume via `current_step`
    """

    def __init__(
        self,
        optimizer_params,
        total_steps: int,
        muon_warmup: int = 1000,
        muon_floor: float = 0.0003,
        muon_momentum_schedule: Optional[
            list
        ] = None,  # e.g. [(0,0.95),(1500,0.93),(4000,0.91)]
        last_step: int = -1,
    ):
        self.optimizer_params = optimizer_params
        self.total_steps = max(1, total_steps)
        self.muon_warmup = muon_warmup
        self.muon_floor = muon_floor

        self.muon_momentum_schedule = muon_momentum_schedule or [(0, 0.95)]

        self.current_step = last_step

    def _cosine_scale(self, step, warmup, floor):
        if step < warmup:
            return (step + 1) / max(1, warmup)

        effective = step - warmup
        remain = max(1, self.total_steps - warmup)
        prog = min(1.0, effective / remain)

        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))

    def step(self):
        self.current_step += 1
        step = self.current_step

        # Determine muon momentum for this step (largest threshold <= step)
        momentum = None
        for threshold, val in self.muon_momentum_schedule:
            if step >= threshold:
                momentum = val
            else:
                break

        scale = self._cosine_scale(step, self.muon_warmup, self.muon_floor)
        self.optimizer_params["lr"] = self.optimizer_params["lr"] * scale

        if momentum is not None:
            self.optimizer_params["momentum"] = momentum

    def state_dict(self):
        return {
            "current_step": self.current_step,
        }

    def load_state_dict(self, state):
        self.current_step = state.get("current_step", self.current_step)
