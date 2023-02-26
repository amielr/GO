import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from mag_algorithms.pso import pso
from scan_tag_gui import ScanTagGui
from mag_utils.mag_utils.scans import HorizontalScan
from pso_class import ParticleSwarmsSimulation
from pso_gui import build_gui_pso, get_bounds_by_coords, cut_df_by_coords
from simulations.dipole_sims import create_magnetic_dipole_simulation as simulate


def calculate_sim_and_loss_from_parameters(mag_scan, pso_results):
    scan_b = mag_scan.b - mag_scan.b.mean()
    simulated_b = simulate(np.array([pso_results['x'], pso_results['y'], mag_scan.a.mean() - pso_results['d2s']]).T,
                           pso_results['measurement'],
                           np.array([mag_scan.x, mag_scan.y, mag_scan.a]).T)
    simulated_b -= simulated_b.mean()
    simulated_b = simulated_b * 1e9
    scan_loss = np.linalg.norm(scan_b - simulated_b)
    return simulated_b, scan_loss


def save_pso_results(pso_results: dict, mag_scan: HorizontalScan, pso_swarm: ParticleSwarmsSimulation, save_path=None,
                     coords=None):
    """

    Args:
        pso_results: the (x,y,z,mx,my,mz) of the dipole values of the final pso result
        mag_scan: HorizontalScan data of the scan you did pso on
        pso_swarm: the pso object that contains the answers of all the particles and the pso params
        save_path: save file destination (.text)
        coords: the [x,y] coords of the chosen cut rectangle sacn

    Returns: the save path in which the result's excel is saved

    """

    if not save_path:
        root_path = Path(mag_scan.file_name)
        save_path = root_path.parent.joinpath(
            root_path.stem + 'pso_sim_' + str(pso_swarm.n_particles) + 'particles.csv')
    if coords is None:
        coords = np.zeros((2, 5))

    rect_bounds = np.stack(coords, axis=1)
    scan_loss = [calculate_sim_and_loss_from_parameters(mag_scan, pso_results)[1]] * pso_swarm.n_sources
    excel_data_dict = {'x': pso_results['x'], 'y': pso_results['y'], 'z': mag_scan.a.mean() - pso_results['d2s'],
                       'd2s': pso_results['d2s'], 'mx': pso_results['measurement'][:, 0],
                       'my': pso_results['measurement'][:, 1], 'mz': pso_results['measurement'][:, 2],
                       'M': np.linalg.norm(pso_results['measurement']), 'loss': scan_loss, 'bounds': [rect_bounds]}

    save_df = pd.DataFrame(excel_data_dict, index=None)
    save_df.to_csv(save_path)
    print(f"Saved results to: {save_path}")

    return save_path


def plot_pso(mag_scan: HorizontalScan, pso_results: dict):
    simulated_b, _ = calculate_sim_and_loss_from_parameters(mag_scan, pso_results)
    mag_scan.b = mag_scan.b - mag_scan.b.mean()
    simulated_scan = HorizontalScan(file_name='simulated_scan', x=mag_scan.x, y=mag_scan.y, a=mag_scan.a,
                                    b=simulated_b, time=mag_scan.time,
                                    is_base_removed=mag_scan.is_base_removed)

    subtracted_mag_scan = deepcopy(mag_scan)
    subtracted_mag_scan.b = subtracted_mag_scan.b - simulated_b

    mag_scan.interpolate('Linear', 0.5, True)
    simulated_scan.interpolate('Linear', 0.5, True)
    subtracted_mag_scan.interpolate('Linear', 0.5, True)

    fig = plt.figure()
    real_scan_ax = fig.add_subplot(221)
    simulated_scan_ax = fig.add_subplot(222)
    subtracted_scan_ax = fig.add_subplot(223)

    axes = [real_scan_ax, simulated_scan_ax, subtracted_scan_ax]
    vmin, vmax = np.min(mag_scan.b), np.max(mag_scan.b)

    real_scan_ax = axes[0].tricontourf(mag_scan.x, mag_scan.y, mag_scan.b, levels=80)
    axes[0].tricontour(mag_scan.x, mag_scan.y, mag_scan.b, levels=30, colors="black", linewidths=0.2)
    axes[0].tricontour(mag_scan.x, mag_scan.y, mag_scan.b, levels=6, colors="black", linewidths=0.5)
    axes[0].set_title("Real Scan")
    plt.colorbar(real_scan_ax).ax.set_ylabel("[nT]", loc="top")

    simulated_scan_ax = axes[1].tricontourf(mag_scan.x, mag_scan.y, simulated_b, levels=80)
    axes[1].tricontour(mag_scan.x, mag_scan.y, simulated_b, levels=30, colors="black", linewidths=0.2)
    axes[1].tricontour(mag_scan.x, mag_scan.y, simulated_b, levels=6, colors="black", linewidths=0.5)
    axes[1].set_title("Simulated Scan")
    plt.colorbar(simulated_scan_ax).ax.set_ylabel("[nT]", loc="top")

    subtracted_scan_ax = axes[2].tricontourf(mag_scan.x, mag_scan.y, subtracted_mag_scan.b, levels=80, vmin=vmin,
                                             vmax=vmax)
    axes[2].tricontour(mag_scan.x, mag_scan.y, subtracted_mag_scan.b, levels=30, colors="black", linewidths=0.2)
    axes[2].tricontour(mag_scan.x, mag_scan.y, subtracted_mag_scan.b, levels=6, colors="black", linewidths=0.5)
    axes[2].set_title("Subtracted Real and Simulated Scan")
    plt.colorbar(subtracted_scan_ax).ax.set_ylabel("[nT]", loc="top")

    for ax in axes:
        ax.scatter(pso_results['x'], pso_results['y'], marker="X", c="red", alpha=0.5, edgecolor='black')
    fig.suptitle('PSO results ' + mag_scan.file_name.split('\\')[-1][:-4])
    plt.show()

    return fig


def get_cut_scan_parameters(mag_scan: HorizontalScan, bounds_type: str, gui_obj: build_gui_pso, tag):
    # change parameters by whats entered by user
    if bounds_type == 'cutscan':
        mag_scan = cut_df_by_coords(mag_scan, gui_obj.x_coord, gui_obj.y_coord)
        bounds = get_bounds_by_coords(mag_scan, gui_obj.x_coord, gui_obj.y_coord, gui_obj.m_max,
                                      gui_obj.n_sources)
    elif bounds_type == 'bounds':
        bounds = get_bounds_by_coords(mag_scan, gui_obj.x_coord, gui_obj.y_coord, gui_obj.m_max,
                                      gui_obj.n_sources)
    else:
        m_max = gui_obj.m_max
        if m_max is None:
            # deltaB of the scan * r^3 * mu0/(4pi)
            m_max = (mag_scan.b.max() - mag_scan.b.min()) * 10 ** -2 * np.linalg.norm(
                [mag_scan.x[mag_scan.b.argmax()] - mag_scan.x[mag_scan.b.argmin()],
                 mag_scan.y[mag_scan.b.argmax()] - mag_scan.y[mag_scan.b.argmin()]]) ** 3

        normalization_factor = 1 / np.sqrt(3) * 1.5
        m_normalized = m_max * normalization_factor
        lower_bound = [mag_scan.x.min(), mag_scan.y.min(), mag_scan.a.mean() - 35, -m_normalized, -m_normalized,
                       -m_normalized]
        upper_bound = [mag_scan.x.max(), mag_scan.y.max(), mag_scan.a.mean(), m_normalized, m_normalized, m_normalized]
        bounds = (lower_bound * gui_obj.n_sources, upper_bound * gui_obj.n_sources)

    scan_tag = tag  # widow or ish

    if scan_tag == 'widow':
        mag_scan.a -= 5
    elif scan_tag == 'ia':
        pass
    else:
        pass

    return mag_scan, bounds

def get_scan_tag_gui():
    scan_tag_obj = ScanTagGui()
    scan_tag_obj.build_window()
    return scan_tag_obj.tag

def pso_running_function(tag):

    # create window with rectangle selector and params
    pso_gui_object = build_gui_pso()

    scan = pso_gui_object.scan
    origin_scan = deepcopy(scan)

    # create necessary parameters
    scan_path = pso_gui_object.scan_path

    scan, bounds = get_cut_scan_parameters(scan, pso_gui_object.bounds_or_cutscan, pso_gui_object, tag=tag)

    swarm = pso.ParticleSwarmOptimization(n_sources=pso_gui_object.n_sources,
                                          bounds=bounds, pso_iterations=pso_gui_object.n_loops,
                                          n_iterations=pso_gui_object.n_iterations,
                                          n_particles=pso_gui_object.n_particles,
                                          options=pso_gui_object.options,
                                          ftol=pso_gui_object.ftol, ftol_iter=pso_gui_object.ftol_iter)

    a = swarm.run(scan, False)

    if pso_gui_object.isplot:
        plot_pso(mag_scan=origin_scan, pso_results=a)

    if pso_gui_object.isecxel:
        coords = [pso_gui_object.x_coord, pso_gui_object.y_coord]
        save_pso_results(pso_results=a, mag_scan=scan, pso_swarm=swarm, coords=coords)
        pass
if __name__ == "__main__":
    tag = get_scan_tag_gui()
    pso_running_function(tag)