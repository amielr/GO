from mag_utils.mag_utils import load
from mag_algorithms.mag_algorithms.pso.pso import ParticleSwarmOptimization
from mag_algorithms.mag_algorithms.euler_deconvolution.euler_deconvolution import EulerDeconvolution

scan = load(r"C:\Users\97252\Desktop\GZ_20F1000_MESURE_20220804064138_polygon_A_6.5_meter.txt")
scan.interpolate("Nearest", 0.5, inplace=True)
# after interpolation there are two places where the data is stored
# scan.data and scan.interpolated_data

scan.interpolated_data.plot(levels=10)

# pso = ParticleSwarmOptimization(n_particles=500, verbose=True)
# out = pso.run(scan)
# print(out)
