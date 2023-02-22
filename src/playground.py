import numpy as np

from mag_utils.mag_utils import load
from mag_algorithms.pso.pso import ParticleSwarmOptimization
from matplotlib import pyplot as plt
import time


scan = load(r"C:\Users\ruti1\Documents\ish\data\GO_data\GZ_20F1000_MESURE_20220804064138_polygon_A_6.5_meter.txt")
scan.interpolate("Nearest", 1, inplace=True)

scan.plot()


"""
A check to see how long it takes to run PSO with different n_particles and n_iterations.
"""
n_iterations = [10, 20]
n_particels = [10, 20, 40]
time_matrix = np.zeros((len(n_iterations), len(n_particels)))

for i, iter in enumerate(n_iterations):
    for j, part in enumerate(n_particels):
        pso = ParticleSwarmOptimization(n_particles=part, n_iterations=iter)
        start_time = time.time()
        out = pso.run(scan)
        run_time = time.time() - start_time
        time_matrix[i, j] = run_time

plt.imshow(time_matrix, cmap='gray')
plt.yticks(list(range(len(n_iterations))), n_iterations)
plt.xticks(list(range(len(n_particels))), n_particels)
plt.colorbar()
plt.xlabel('particles')
plt.ylabel('iterations')
plt.show()



