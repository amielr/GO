from simulated_scan import SimulatedScan
from magnetic_objects import ConstantField, MagneticDipole, FiniteDipoleChain
import matplotlib.pyplot as plt
import numpy as np



def show_mag_anachi_simulation(sources_list, xy_list, z_end=-50, res=0.1, b_ext=ConstantField()):
    """
    for simulating magnetic field in boreholes

    :param sources_list: list of magnetic objects
    :param xy_list: list of [x,y] of the boreholes [m] (for example, [[1, 2], [1, 3], [2, 3]])
    :param z_end: end of the boreholes
    :param res: sample res [s]
    :param b_ext: ConstantField() object
    :return: Scan object, matrix of simulated magnetic fields in boreholes
    """

    scans = SimulatedScan()
    scans.add_source(b_ext)
    z_end = -z_end * np.sign(z_end)  # make sure it's below zero
    res = -res * np.sign(res)
    xy_list = np.array(xy_list)
    for i in range(len(xy_list)):
        scans.add_route(np.array([[xy_list[i][0], xy_list[i][1], z] for z in np.arange(0, z_end, res)]))
    for source in sources_list:
        scans.add_source(source)
    magnetic_field_matrix = scans.calculate_magnetic_field(True)
    for i, mag in enumerate(magnetic_field_matrix):
        magnetic_field_matrix[i] = (mag - np.mean(mag)) * 10 ** 9  # around zero and in [nT]
    return scans, magnetic_field_matrix


def plot_8_boreholes(scan_obj, magnetic_filed_matrix):
    fig = plt.figure()
    z_array = scan_obj.scan_route[0][:, 2]
    geo_ax = fig.add_subplot(221)

    all_ax = fig.add_subplot(448)
    all_ax.set_xticklabels([])
    all_ax.grid(which='both')
    all_ax.minorticks_on()
    all_without_first_ax = fig.add_subplot(447)
    all_without_first_ax.set_xticklabels([])
    all_without_first_ax.grid(which='both')
    all_without_first_ax.minorticks_on()
    plot_ax = [fig.add_subplot(4, 4, i) for i in np.arange(9, 17)]

    for i, ax in enumerate(plot_ax):
        if i < 4:
            all_ax.plot(-z_array, magnetic_filed_matrix[3 - i], label=3 - i + 1)
            if i - 3:
                all_without_first_ax.plot(-z_array, magnetic_filed_matrix[3 - i], label=3 - i + 1)
        ax.plot(-z_array, magnetic_filed_matrix[i], linewidth=2, c='black')
        ax.minorticks_on()
        ax.grid(which='both')
        ax.set_title("borehole " + str(i + 1))

        if i > 3:
            ax.set_xlabel('Depth [m]')
        else:
            ax.set_xticklabels([])
    all_ax.legend(loc='lower left')
    all_without_first_ax.legend()
    plot_ax[0].set_ylabel('Magnetic Field [nT]')
    plot_ax[4].set_ylabel('Magnetic Field [nT]')
    all_without_first_ax.set_ylabel('Magnetic Field [nT]')
    all_without_first_ax.set_title('Boreholes 2-4')
    all_ax.set_title('Boreholes 1-4')

    for source in scan_obj.scan_sources:
        source_pos = source.get_pos()
        if source_pos.shape == (3,):
            geo_ax.scatter(source_pos[0], source_pos[1])
        else:
            geo_ax.scatter(source_pos[:, 0], source_pos[:, 1], label='source')

    for i, borehole in enumerate(scan_obj.scan_route):
        if i == 1 and str(type(source)) != '<class \'__main__.ConstantField\'>':
            geo_ax.scatter(borehole[0, 0], borehole[0, 1], c='black', label='borehole')
        else:
            geo_ax.scatter(borehole[0, 0], borehole[0, 1], c='black')
        # geo_ax.text(borehole[0, 0] - 0.5, borehole[0, 1] + 0.5, "borehole " + str(i + 1), c='black')
        geo_ax.text(borehole[0, 0] + 0.5, borehole[0, 1] + 0.5, str(i + 1), c='black')
    geo_ax.axis('equal')
    geo_ax.set_xlabel('East [m]')
    geo_ax.set_ylabel('North [m]')
    geo_ax.set_xlim([geo_ax.get_xlim()[0] - 2, geo_ax.get_xlim()[1] + 2])
    geo_ax.set_ylim([geo_ax.get_ylim()[0] - 2, geo_ax.get_ylim()[1] + 2])
    geo_ax.legend()
    geo_ax.grid(which='both')
    fig.suptitle('Boreholes mag Simulation', size=15)
    b_ext = (scan_obj.scan_sources[0].calculate_magnetic_field([0]) * 10 ** 9)[0]
    b_ext_norm = np.linalg.norm(b_ext)

    fig.text(0.52, 0.87, 'External Magnetic Field = ' + str(list(np.round(b_ext / b_ext_norm, 3))) + ' * ' + str(
        np.round(b_ext_norm, 2)) + ' [nT]')
    for i, source in enumerate(scan_obj.scan_sources):
        if str(type(source)) != '<class \'Create_Scans.ConstantField\'>':
            fig.text(0.52, 0.83 - 0.2 * (i - 1), 'Moment = ' + str(list(source.moment)) + ' ' + source.moment_type)
            fig.text(0.52, 0.79 - 0.2 * (i - 1),
                     'Min Depth = ' + str(np.min(source.get_pos()[:, 2])) + ' [m]' + '     ' +
                     'Max Depth = ' + str(np.max(source.get_pos()[:, 2])) + ' [m]')
            if str(type(source)) == '<class \'Create_Scans.FiniteDipoleChain\'>':
                fig.text(0.52, 0.75 - 0.2 * (i - 1), 'Density = ' + str(source.density) + ' [dipoles/m]')

    plt.show()


def show_4_widow_graphs(scan_obj):
    fig, axes = plt.subplots(2, 2, sharex=False, sharey=False)
    axes = axes.flatten()
    b = scan_obj.calculate_magnetic_field(True)
    routes = scan_obj.scan_route
    for i, ax in enumerate(axes):
        cm = ax.tricontourf(routes[i][:, 0], routes[i][:, 1], (b[i] - b[i].mean()) * 10 ** 9, levels=40)
        ax.tricontour(routes[i][:, 0], routes[i][:, 1], (b[i] - b[i].mean()) * 10 ** 9, levels=40, colors='black',
                      linewidths=0.2)
        ax.tricontour(routes[i][:, 0], routes[i][:, 1], (b[i] - b[i].mean()) * 10 ** 9, levels=8, colors='black',
                      linewidths=0.5)
        for source in scan_obj.scan_sources:
            source_pos = source.get_pos()
            if source_pos.shape == (3,):
                ax.scatter(source_pos[0], source_pos[1])
            else:
                ax.scatter(source_pos[:, 0], source_pos[:, 1], label='source')
        ax.set_title('flight height = ' + str(routes[i][:, 2].mean()) + ' meters above object')
        ax.set_xlabel('East [m]')
        ax.set_ylabel('North [m]')
        ax.axis('equal')
        fig.colorbar(cm, ax=ax, label='Magnetic Field [nT]')
    fig.suptitle('widow Simulation For Several Heights')
    plt.show()


def subtract_2_widow_scans(magnetic_objects_list, x_lim, y_lim, z0, z1, b_external=ConstantField(), plot = True):
    scan = SimulatedScan()
    scan.add_source(b_external)
    scan.add_source(magnetic_objects_list)
    scan.add_rectangular_route(4, 10, -x_lim, x_lim, -y_lim, y_lim, z0, 60, 0)
    scan.add_rectangular_route(4, 10, -x_lim, x_lim, -y_lim, y_lim, z1, 60, 0)

    b = scan.calculate_magnetic_field(True)
    xs, ys = scan.scan_route[0][:, 0], scan.scan_route[0][:, 1]
    for i, _ in enumerate(b):
        b[i] = b[i] - np.mean(b[i])
    fig1, axs1 = plt.subplots(2, 2)
    axs1 = axs1.flatten()

    sources_locations = scan.get_sources_pos()
    cm0 = axs1[0].tricontourf(xs, ys, b[0], levels=30)
    cm1 = axs1[1].tricontourf(xs, ys, b[1], levels=30)
    cm2 = axs1[2].tricontourf(xs, ys, b[0] - b[1], levels=30)

    axs1[0].scatter(sources_locations[:, 0], sources_locations[:, 1])
    axs1[1].scatter(sources_locations[:, 0], sources_locations[:, 1])
    axs1[2].scatter(sources_locations[:, 0], sources_locations[:, 1])

    axs1[0].set_title('height = ' + str(scan.scan_route[0][0][2]))
    axs1[1].set_title('height = ' + str(scan.scan_route[1][0][2]))
    axs1[2].set_title('subtraction')
    fig1.colorbar(cm0, ax=axs1[0])
    fig1.colorbar(cm1, ax=axs1[1])
    fig1.colorbar(cm2, ax=axs1[2])

    if plot:
        plt.show()
    return b, fig1


if __name__ == '__main__':
    test_scan = SimulatedScan()
    B_earth = ConstantField()
    tunnel = FiniteDipoleChain(p_initial=np.array([-20, -20, 0]), p_final=np.array([-20, 60, 0]),
                               m=[0, 20, -20], density=2)  # horizontal
    # test_scan.add_source(tunnel)
    d1 = MagneticDipole(p_vec=[0,0,0],m_vec=[0,20,-20])
    test_scan.add_source(d1)
    test_scan.add_source(B_earth)
    route = add_rectangular_route(v=5, fs=10, x0=-100, x1=100, y0=-100, y1=100, z=20, num_lines=100)
    for i in range (0,20):
        d2 = MagneticDipole(p_vec=[0, 30, 0], m_vec= [0, 20, -20])
        test_scan.add_source(d2)
        test_scan.add_route(route)
        #test_scan.remove_source(2)
    test_scan.create_fig(1, 1, 1)

    # tunnel = FiniteDipoleChain(p_initial=np.array([-20, -20, -30]), p_final=np.array([20, 20, -30]),
    #                            m=[0, 20, -20], density=10)  # alachson

    # tunnel = FiniteDipoleChain(p_initial=np.array([0, -20, -30]), p_final=np.array([0, 20, -30]),
    #                            m=[0, 20, -20], density=10)  # vertical
    # d1 = MagneticDipole(p_vec=[0, 0, 0], m_vec=[0, 10, -10])
    # d2 = MagneticDipole(p_vec=[10, 10, 0], m_vec=[0, 10, -10])
    # boreholes_list = [[0, -1], [0, -2], [0, -4], [0, -8], [0, 1], [0, 2], [0, 4], [0, 8]]  # vertical

    # B_ext = ConstantField()
    # b, fig1 = subtract_2_widow_scans([d1, d2], 40, 40, 10, 12, plot=False)
    # fig1.text(0.55, 0.43, 'd1 Moment = ' + str(list(d1.moment)) + ' ' + d1.moment_type)
    # fig1.text(0.55, 0.35, 'd2 Moment = ' + str(list(d2.moment)) + ' ' + d2.moment_type)

    # fig1.text(0.55, 0.27, 'd1 location = ' + str(list(d1.get_pos())))
    # fig1.text(0.55, 0.19, 'd2 location = ' + str(list(d2.get_pos())))

    # for i, ax1 in enumerate(axs1.flatten()):
    #    ax1.tricontourf(xy[:, 0], xy[:, 1], b[i+1] - b[0])
    #
    # fig2, axs2 = plt.subplots(2, 2)
    # for i, ax2 in enumerate(axs2.flatten()):
    #     ax2.tricontourf(xy[:, 0], xy[:, 1], b[i+1])

    plt.show()
