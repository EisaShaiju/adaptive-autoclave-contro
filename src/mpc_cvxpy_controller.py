"""Baseline controller interface used by higher-level scripts.

The implementation intentionally keeps CVXPY optional so the module can be
imported in environments where the solver stack is not installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MPCCVXPyController:
    """Lightweight fallback controller with a stable API."""

    target_temperature: float
    proportional_gain: float = 10.0
    min_power: float = 0.0
    max_power: float = 10000.0

    def compute_control(self, current_temperature: float) -> float:
        error = self.target_temperature - float(current_temperature)
        power = self.proportional_gain * error
        return max(self.min_power, min(self.max_power, power))
