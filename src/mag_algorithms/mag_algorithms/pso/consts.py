"""consts."""
import numpy as np

B_EXT = np.array([0, 1, -1]) / np.sqrt(2) * 4.4 * 1e-5  # [T]
MU0 = 10 ** -7 * (4 * np.pi)
#LOWER_BOUND_OFFSET = [-20, -20, -60]
#UPPER_BOUND_OFFSET = [20, 20, -1]
LOWER_BOUND_OFFSET = [-0., -0., -60]
UPPER_BOUND_OFFSET = [-0., -0., -1]

"""
dict with pso parameter {'c1', 'c2', 'w'}
a dictionary containing the parameters for the specific
optimization technique.
    * c1 : float
         cognitive parameter
    * c2 : float
        social parameter
    * w : float
        inertia parameter
"""
OPTIONS = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
