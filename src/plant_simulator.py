"""Simple autoclave plant model used by baseline tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class AutoclavePlant:
    """A minimal first-order thermal model for autoclave temperature dynamics."""

    ambient_temperature: float = 25.0
    initial_temperature: float = 25.0
    thermal_mass: float = 1000.0
    heat_loss_coeff: float = 0.1
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.thermal_mass <= 0:
            raise ValueError("thermal_mass must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        self.temperature = float(self.initial_temperature)

    def reset(self, temperature: float | None = None) -> float:
        self.temperature = float(self.initial_temperature if temperature is None else temperature)
        return self.temperature

    def step(self, heater_power: float) -> float:
        net_power = float(heater_power) - self.heat_loss_coeff * (self.temperature - self.ambient_temperature)
        self.temperature += (net_power / self.thermal_mass) * self.dt
        return self.temperature

    def simulate(self, heater_profile: Iterable[float]) -> List[float]:
        return [self.step(power) for power in heater_profile]
