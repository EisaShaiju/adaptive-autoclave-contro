import unittest

from src.mpc_cvxpy_controller import MPCCVXPyController
from src.plant_simulator import AutoclavePlant


class SourceModuleImportTests(unittest.TestCase):
    def test_autoclave_plant_is_importable_and_steps(self):
        plant = AutoclavePlant(initial_temperature=25.0, thermal_mass=1000.0, dt=1.0)
        next_temp = plant.step(1000.0)
        self.assertGreater(next_temp, 25.0)

    def test_controller_is_importable_and_clamps_power(self):
        controller = MPCCVXPyController(target_temperature=200.0, max_power=500.0)
        power = controller.compute_control(current_temperature=0.0)
        self.assertEqual(power, 500.0)


if __name__ == "__main__":
    unittest.main()
