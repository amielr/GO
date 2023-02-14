from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.interpolate
from matplotlib import pyplot as plt
from osgeo import gdal
from mag_utils.loader.blackwidow import load_gz_scan
from mag_utils.algo_tools.simulations import simulate_b_ax_from_dipole
from mag_utils.algo_tools.convex_hull_grids import convex_hull_grid
from mag_utils.scans import HorizontalScan
from tqdm import tqdm

SPACING = 10


def dsm_tiff_to_3d_points(src: str, spacing: float = SPACING):
    tiff = gdal.Open(src)
    gdal.Translate(src + ".xyz", tiff)
    df = pd.read_csv(src + ".xyz", sep=' ', header=None)
    f = scipy.interpolate.SmoothBivariateSpline(df[0], df[1], df[2])
    xs, ys = convex_hull_grid(df[0], df[1], spacing)
    interp_points = np.array([xs, ys, f(xs, ys, grid=False)]).T
    return interp_points


def get_grid_from_upper_envelope(points: np.ndarray, spacing: float = SPACING):
    z_min = points[:, 2].min()
    new_points = []
    for point in points:
        zs = np.arange(point[2], z_min, -spacing)
        for z in zs:
            new_points.append((point[0], point[1], z))
    return np.array(new_points)


def get_land_simulation_at_scan(land_dipoles, scan, moment, b_e=np.array([0., 1., -1.]), subtract_dc=True):
    sim_vec = np.zeros(len(scan.b))
    scan_xyz = np.array([scan.x, scan.y, scan.a]).T

    for dipole in tqdm(land_dipoles):
        sim_vec += simulate_b_ax_from_dipole(dipole, moment, b_e, scan_xyz)

    if subtract_dc:
        sim_vec -= sim_vec.mean()

    return sim_vec


def clean_scan_by_dsm(scan: HorizontalScan, dsm_tif_path: str, plot_3d_scatter: bool = False):
    scan = deepcopy(scan)

    points = dsm_tiff_to_3d_points(dsm_tif_path)
    points = get_grid_from_upper_envelope(points)

    if plot_3d_scatter:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b')
        ax.scatter(scan.x, scan.y, scan.a, c='r')

    sim_x = get_land_simulation_at_scan(points, scan, np.array([1., 0, 0]))
    sim_y = get_land_simulation_at_scan(points, scan, np.array([0, 1., 0]))
    sim_z = get_land_simulation_at_scan(points, scan, np.array([0, 0, 1.]))

    q = np.linalg.qr(np.array([np.ones_like(scan.b),
                               sim_x,
                               sim_y,
                               sim_z]).T)[0]

    ground_noise = q @ (q.T @ scan.b)
    scan.b -= ground_noise

    return scan, ground_noise


def plot_scan_and_cleaned_scan(scan_path, dsm_tiff_path, plot_3d_scatter=True):
    scan = load_gz_scan(scan_path)

    clean_scan, ground_noise = clean_scan_by_dsm(scan, dsm_tiff_path, plot_3d_scatter=plot_3d_scatter)

    fig, ax = plt.subplots(ncols=1, nrows=3)
    ax[0].tricontourf(scan.x, scan.y, scan.b)
    ax[1].tricontourf(scan.x, scan.y, ground_noise)
    ax[2].tricontourf(clean_scan.x, clean_scan.y, clean_scan.b)
    for a in ax:
        a.axis("equal")

    plt.show()


def main():
    dsm_tiff_path = "<filename.tif>"
    scan_path = "<insert path here>"
    plot_scan_and_cleaned_scan(scan_path, dsm_tiff_path)


if __name__ == '__main__':
    main()
