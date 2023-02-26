import ast
import sys
import numpy as np
import tkinter as tk
from pathlib import Path
import tkinter.filedialog
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mag_utils.mag_utils.scans import HorizontalScan
from mag_utils.mag_utils.loader.main_load import load

g1_path = str(Path(__name__).absolute().parents[1] / "ground_one")
if g1_path not in sys.path:
    sys.path.append(g1_path)

from ground_one.graphics.FilterWindow import FilterWindow


class GuiForPso:  # Gui class for opening a mag scan and choosing a rectangle.

    def __init__(self):
        # initialize scan matrix and scan path
        self.scan = None
        self.scan_path = None

        # create figure
        self.fig = Figure(figsize=(5, 5), dpi=100)

        # tk window
        self.next_window = tk.Tk()
        self.next_window.geometry('850x800+0+0')
        self.next_window.title('Pso Gui')

        # adding a plot to window
        self.plot1 = self.fig.add_subplot(111)
        self.colorbar = None

        # creating the canvas in which the plot is going to be
        self.canvas = None

        # rectangle corners
        self.x_coord = None
        self.y_coord = None
        self.save_params_button = None
        self.instruction_text = None
        self.rs = None

        # parameters of pso
        self.n_particles = None
        self.n_iterations = None
        self.n_loops = None
        self.n_sources = None
        self.m_max = None
        self.bounds_or_cutscan = None
        self.isecxel = None
        self.isplot = None
        self.options = None
        self.ftol = None
        self.ftol_iter = None

        # UI for parameters
        self.radio_button_bounds = None
        self.radio_button_cutscan = None

        self.particles_label = None
        self.particles_entry = None

        self.iterations_label = None
        self.iterations_entry = None

        self.loops_label = None
        self.loops_entry = None

        self.sources_label = None
        self.sources_entry = None

        self.options_label = None
        self.options_entry = None

        self.ftol_label = None
        self.ftol_entry = None
        self.ftol_iter_label = None
        self.ftol_iter_entry = None

        # checks if save or not
        self.excel_checkbox = None
        self.isecxel_var = tk.IntVar()

        # to plot or not
        self.plot_checkbox = None
        self.isplot_var = tk.IntVar()

        # create label and entry for m_max
        self.m_max_label = None
        self.m_max_entry = None

    def get_scan(self):
        self.scan_path = tkinter.filedialog.askopenfilename()
        self.scan = load(self.scan_path)

    def set_scan(self, scan):
        self.scan = scan
        self.draw_graphs()

    def draw_rect_and_scan_on_subplot(self):
        """

        :return: draws the rectancle selector and scan on tkinter window and save parameters save_params_button
        """

        def save_params():
            self.n_particles = int(self.particles_entry.get())
            self.n_iterations = int(self.iterations_entry.get())
            self.n_loops = int(self.loops_entry.get())
            self.n_sources = int(self.sources_entry.get())
            self.options = ast.literal_eval(self.options_entry.get())
            self.ftol = float(self.ftol_entry.get())
            self.ftol_iter = int(self.ftol_iter_entry.get())

            if (self.m_max_entry.get() is not None) and (self.m_max_entry.get() != 'None'):
                self.m_max = int(self.m_max_entry.get())

            # if save save_params_button is pressed or not
            self.isecxel = self.isecxel_var.get()
            self.isplot = self.isplot_var.get()

            self.x_coord, self.y_coord = self.rs.geometry[1, :], self.rs.geometry[0, :]
            self.next_window.destroy()

        def open_new_file():
            self.get_scan()
            self.draw_graphs()

        self.instruction_text = tk.Label(self.next_window,
                                         text="mark the indication you would like to calculate\n by clicking and dragging the mouse",
                                         width=40, height=3, font=('David', 20))
        self.instruction_text.place(x=0, y=0)

        # create canvas
        self.draw_graphs()

        self.save_params_button = tk.Button(master=self.next_window, text="save coords and settings",
                                            command=save_params)
        self.save_params_button.place(x=100, y=650)

        self.new_file_button = tk.Button(master=self.next_window, text="choose new file", command=open_new_file)
        self.new_file_button.place(x=650, y=650)

        self.filter_button = tk.Button(master=self.next_window, text="Filter Scan", command=lambda: FilterWindow(self))
        self.filter_button.place(x=650, y=470)

    def draw_parameters_entries(self):
        """

        :return: creates a label and entry for the relevant parameters for PSO
        particles --- the number of different sources to simulate
        iterations --- the number of iteration in which you create new particles and act accordingly
        loops --- the number of times you run a optimizer with the particles and iterations
        """

        def set_radio_button_flag():
            if var.get() == 1:
                self.bounds_or_cutscan = "bounds"

            if var.get() == 2:
                self.bounds_or_cutscan = "cutscan"

        # for particles
        self.particles_label = tk.Label(master=self.next_window, text="particles")
        self.particles_label.place(x=650, y=100)

        self.particles_entry = tk.Entry(master=self.next_window)
        self.particles_entry.insert(0, string='200')
        self.particles_entry.place(x=650, y=120)

        # for iterations
        self.iterations_label = tk.Label(master=self.next_window, text="iterations")
        self.iterations_label.place(x=650, y=150)

        self.iterations_entry = tk.Entry(master=self.next_window)
        self.iterations_entry.insert(0, string='200')
        self.iterations_entry.place(x=650, y=170)

        # for number of loops
        self.loops_label = tk.Label(master=self.next_window, text="loops")
        self.loops_label.place(x=650, y=200)

        self.loops_entry = tk.Entry(master=self.next_window)
        self.loops_entry.insert(0, string='1')
        self.loops_entry.place(x=650, y=220)

        # for number of sources
        self.sources_label = tk.Label(master=self.next_window, text="sources")
        self.sources_label.place(x=650, y=250)

        self.sources_entry = tk.Entry(master=self.next_window)
        self.sources_entry.insert(1, string='1')
        self.sources_entry.place(x=650, y=270)

        # for changing stop condition of pso
        self.ftol_label = tk.Label(master=self.next_window, text="ftol")
        self.ftol_label.place(x=650, y=300)

        self.ftol_entry = tk.Entry(master=self.next_window)
        self.ftol_entry.insert(0, "1e-10")
        self.ftol_entry.place(x=650, y=320)

        self.ftol_iter_label = tk.Label(master=self.next_window, text="ftol iter")
        self.ftol_iter_label.place(x=650, y=350)

        self.ftol_iter_entry = tk.Entry(master=self.next_window)
        self.ftol_iter_entry.insert(0, "40")
        self.ftol_iter_entry.place(x=650, y=370)

        # for changing m_max (the moment bounds of pso)
        self.m_max_label = tk.Label(master=self.next_window, text="m-max")
        self.m_max_label.place(x=650, y=500)

        self.m_max_entry = tk.Entry(master=self.next_window)
        self.m_max_entry.insert(0, string='None')
        self.m_max_entry.place(x=650, y=520)

        # for selecting between bound choosing, scan cutting or default
        var = tk.IntVar()
        self.radio_button_bounds = tk.Radiobutton(master=self.next_window, text='Select by bounds', variable=var,
                                                  value=1, command=set_radio_button_flag)
        self.radio_button_bounds.place(x=650, y=550)

        self.radio_button_cutscan = tk.Radiobutton(master=self.next_window, text='Select by cutting scan', variable=var,
                                                   value=2, command=set_radio_button_flag)
        self.radio_button_cutscan.place(x=650, y=570)

        self.radio_button_cutscan = tk.Radiobutton(master=self.next_window, text='default scan', variable=var, value=3,
                                                   command=set_radio_button_flag)
        self.radio_button_cutscan.select()
        self.radio_button_cutscan.place(x=650, y=590)

        # create is save excel checkbutton
        self.excel_checkbox = tk.Checkbutton(master=self.next_window, text='save excel', variable=self.isecxel_var)
        self.excel_checkbox.place(x=300, y=650)

        # create isplot checkbutton
        self.plot_checkbox = tk.Checkbutton(master=self.next_window, text='plot', variable=self.isplot_var)
        self.plot_checkbox.toggle()
        self.plot_checkbox.place(x=400, y=650)

        # for changing options
        self.options_label = tk.Label(master=self.next_window, text="options:")
        self.options_label.place(x=300, y=700)
        self.options_entry = tk.Entry(master=self.next_window)
        self.options_entry.insert(0, "{'c1':0.3 ,'c2':0.5 ,'w':0.9}")
        self.options_entry.place(x=360, y=700)

    def draw_graphs(self):
        self.fig.clear()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.next_window)
        self.canvas.draw()
        toolbar = NavigationToolbar2Tk(self.canvas, self.next_window)
        toolbar.place(x=100, y=600)
        self.canvas.get_tk_widget().place(x=100, y=100)
        self.ax = self.fig.gca()
        # plot scan
        cm = self.ax.tricontourf(self.scan.x, self.scan.y, self.scan.b, levels=80)
        self.ax.tricontour(self.scan.x, self.scan.y, self.scan.b, colors='black', levels=15,
                           linewidths=1, alpha=0.5)
        self.colorbar = self.fig.colorbar(cm, label='Magnetic Field [nT]')
        # init rectangle selector
        self.rs = RectangleSelector(self.ax, onselect=lambda e1, e2: None, drawtype='box',
                                    useblit=False, button=[1],
                                    minspanx=5, minspany=5, spancoords='pixels', interactive=True)


def build_gui_pso():
    """
    :return: GuiForPso type after choosing a rectangle and confirming
    """
    pso_gui_object = GuiForPso()
    pso_gui_object.get_scan()
    pso_gui_object.draw_rect_and_scan_on_subplot()
    pso_gui_object.draw_parameters_entries()
    tk.mainloop()

    return pso_gui_object


def indexes_to_keep(mag_scan: HorizontalScan, x_coord, y_coord):
    """
    get indexes of mag_scan to keep after cutting it with matplotlib RectangleSelector

    Args:
        mag_scan:
        x_coord:
        y_coord:

    Returns: indexes to keep
    """

    y_indexes_to_keep = np.where((mag_scan.y > y_coord[0]) & (mag_scan.y < y_coord[2]))
    x_indexes_to_keep = np.where((mag_scan.x > x_coord[0]) & (mag_scan.x < x_coord[1]))
    return np.intersect1d(y_indexes_to_keep, x_indexes_to_keep)


def cut_df_by_coords(mag_scan: HorizontalScan, x_coord, y_coord):
    """
    :param df:the full magScan of scan
    :param x_coord: the list of x coords (corners) to cut
    :param y_coord: the list of y coords (corners) to cut
    :return: df cut by the corners
    """

    indexes = indexes_to_keep(mag_scan, x_coord, y_coord)
    if len(indexes):  # not empty
        mag_scan = mag_scan[indexes]

    return mag_scan


def get_bounds_by_coords(mag_scan: HorizontalScan, x_coord, y_coord, m_max=None, n_sources=1):
    """
    :param mag_scan:the full df of scan
    :param x_coord: the list of x coords (corners) for setting bounds
    :param y_coord: the list of y coords (corners) for setting bounds
    :param m_max: m bound [Am^2]
    :return: df cut by the corners

    """

    indexes = indexes_to_keep(mag_scan, x_coord, y_coord)
    if not len(indexes):
        indexes = [i for i in range(len(mag_scan))]  # keep all indexes
    # check if give m_max
    if m_max is None:
        # deltaB of the scan * r^3 * mu0/(4pi)
        m_max = (mag_scan.b.max() - mag_scan.b.min()) * 10 ** -2 * np.linalg.norm(
            [mag_scan.x[mag_scan[indexes].b.argmax()] - mag_scan.x[mag_scan[indexes].b.argmin()],
             mag_scan.y[mag_scan[indexes].b.argmax()] - mag_scan.y[mag_scan[indexes].b.argmin()]]) ** 3

    normalization_factor = 1 / np.sqrt(3) * 1.5
    m_normalized = m_max * normalization_factor

    if len(indexes_to_keep(mag_scan, x_coord, y_coord)):
        lower_bound = [x_coord[0], y_coord[0], mag_scan.a.mean() - 35, -m_normalized, -m_normalized, -m_normalized]
        upper_bound = [x_coord[2], y_coord[2], mag_scan.a.mean(), m_normalized, m_normalized, m_normalized]
    else:  # bounds are not logical
        lower_bound = [mag_scan.x.min(), mag_scan.y.min(), mag_scan.a.mean() - 35, -m_normalized, -m_normalized,
                       -m_normalized]
        upper_bound = [mag_scan.x.max(), mag_scan.y.max(), mag_scan.a.mean(), m_normalized, m_normalized, m_normalized]
    bounds = (lower_bound * n_sources, upper_bound * n_sources)
    return bounds


def get_difference_b_in_cut_scan(cut_scan):
    """
    :param cut_scan: df of the scan
    :return: the difference between b_max and b_min
    """
    if 'original B' not in cut_scan.columns:
        return cut_scan['B'].max() - cut_scan['B'].min()
    else:
        return cut_scan['original B'].max() - cut_scan['original B'].min()


def get_dist_b_in_cut_scan(cut_scan):
    """
    :param cut_scan:(df) the scan after cutting it with rectngle selector
    :return: the 2d distance between the max B and min B
    """
    minB_idx = cut_scan['B'].idxmin()
    maxB_idx = cut_scan['B'].idxmax()

    min_x_of_b = cut_scan['x'].iloc[minB_idx]
    min_y_of_b = cut_scan['y'].iloc[minB_idx]

    max_x_of_b = cut_scan['x'].iloc[maxB_idx]
    max_y_of_b = cut_scan['y'].iloc[maxB_idx]

    dist = np.sqrt((max_x_of_b - min_x_of_b) ** 2 + (max_y_of_b - min_y_of_b) ** 2)
    return dist


if __name__ == '__main__':
    r = build_gui_pso()
    print("to save or not:", r.isecxel)
    print("to cut or use bounds or as usual:", r.bounds_or_cutscan)
    print("m_max:", r.m_max)
    print("sources:", r.n_sources)
    print("plot:", r.isplot)
    print("options:", r.options)
    print("ftol:", r.ftol)
    print("ftol_entry:", r.ftol_iter)

    bounds = get_bounds_by_coords(r.scan, r.x_coord, r.y_coord, r.m_max)
    cut_scan = cut_df_by_coords(r.scan, r.x_coord, r.y_coord)
    print([bounds])

    # useful functions
    # dist = get_dist_b_in_cut_scan(cut_scan)
    # print("dist: ", str(dist))
    # diff = get_difference_b_in_cut_scan(cut_scan)
    # print("diff: "+str(diff))
