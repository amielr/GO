import numpy as np

from mag_utils.algo_tools.sgd_sandbox import SGDSandbox
from ..blackwidow_loader_test import scan_mock


def test_sgd_sandbox(scan_mock):
    dipole_positions = np.array([[0, 0, 0],
                                 [1, 0, 3]])
    coeffs = [1, 2]

    sb = SGDSandbox(scan_mock, dipoles_positions=dipole_positions, init_coeffs=coeffs)

    b_val = sb.simulate_at(np.array([[0, 3, 1]]))

    assert bool(b_val), "Simulation is not true, sandbox is broken ):"
