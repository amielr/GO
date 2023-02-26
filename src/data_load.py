from mag_utils.mag_utils import load
from mag_algorithms.pso.pso import ParticleSwarmOptimization


scan = load(r"C:\Users\97252\Desktop\20F1000_MESURE_20220804062231_CIUL.txt")
scan.plot()

pso = ParticleSwarmOptimization(n_particles=500, verbose=True)
out = pso.run(scan)
print(out)
