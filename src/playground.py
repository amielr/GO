from mag_utils.mag_utils import load
from mag_algorithms.pso.pso import ParticleSwarmOptimization

scan = load(r"C:\Users\ruti1\Documents\ish\data\GO_data\GZ_20F1000_MESURE_20220804064138_polygon_A_6.5_meter.txt")
scan.interpolate("Nearest", 1, inplace=True)

scan.plot()

pso = ParticleSwarmOptimization(n_particles=500)
out = pso.run(scan)
print(out)
