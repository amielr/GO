from tkinter import IntVar, TOP, BOTH, Grid, BOTTOM
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from ttkthemes import ThemedTk
from tkinter.ttk import Frame, Label, Entry, Button, Checkbutton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from magrnd.ground_one.data_processing.consts import GUI_THEME, VOLVO_TITLE, AXES_ASPECT, BG_COLOR, TB_COLOR, TIME_RESOLUTION, \
    ANGLE_TOLERANCE, CLUSTER_DISTANCE, MIN_CLUSTER_SIZE, VOLVO_EXPLANATION, \
    WINDOW_TITLE, LINES_NAME_PREFIX, VOLVO_ENTRY_LABELS
from mag_utils.mag_utils.functional.line_analysis import separate_lines
from magrnd.ground_one.data_processing.magscan2df import convert_magscan_to_df
from tkinter.filedialog import asksaveasfilename
from tkinter.messagebox import showinfo
from pathlib import Path


class VolvoWindow:
    def __init__(self, main_window):
        self.main_window = main_window
        self.volvo_scan = deepcopy(self.main_window.scan)

        self.perpendicular_state = IntVar()
        self.perpendicular_state.set(0)

        self.opposite_state = IntVar()
        self.opposite_state.set(0)

        self.mean_state = IntVar()
        self.mean_state.set(0)

        self.mean_by_dir = IntVar()
        self.mean_by_dir.set(0)

        self.initialize_window()

        # set entry values
        self.set_entry_values()

        self.build_volvo_ui()


        self.run_volvo()
        ################

        # self.switch_mean_state()
        # self.switch_mean_by_dir_state()
        #
        # self.run_volvo()
    #############
    def initialize_window(self):
        self.volvo_window = ThemedTk(themebg=True)
        self.volvo_window.set_theme(GUI_THEME)
        self.volvo_window.wm_title(VOLVO_TITLE)

        # create geographic frame
        self.geo_frame = Frame(self.volvo_window)
        self.geo_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # create and set up settings frame
        self.settings_frame = Frame(self.volvo_window)
        self.settings_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # create checkbox frame
        self.checkboxes_frame = Frame(self.settings_frame)
        self.checkboxes_frame.pack(side=TOP)

        # create entry frame
        self.entry_frame = Frame(self.settings_frame)
        self.entry_frame.pack(side=TOP)

        # create buttons frame
        self.buttons_frame = Frame(self.settings_frame)
        self.buttons_frame.pack(side=TOP)

        # create fig to display interpolation
        self.geo_fig = plt.figure()
        self.geo_axes = self.geo_fig.subplots(2, 1, sharex=True, sharey=True)

        # associate interp fig to canvas
        self.canvas = FigureCanvasTkAgg(self.geo_fig, master=self.geo_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=1)

        # init interp toolbar
        self.geo_toolbar = NavigationToolbar2Tk(self.canvas, self.geo_frame)

        # set theme and aspect

        from magrnd.ground_one.graphics.MainWindow import set_axes_theme  # avoid circular import
        for ax in self.geo_axes:
            ax.set_aspect(AXES_ASPECT)
            set_axes_theme(ax)

        self.canvas.figure.patch.set_facecolor(BG_COLOR)

        self.geo_toolbar.config(background=TB_COLOR)
        self.geo_toolbar._message_label.config(background=TB_COLOR)
        for button in self.geo_toolbar.winfo_children():
            button.config(background=TB_COLOR)

        self.geo_axes[0].scatter(self.volvo_scan.x, self.volvo_scan.y)
        self.canvas.draw()

        # allow window resizing
        Grid.rowconfigure(self.volvo_window, 0, weight=1)
        Grid.columnconfigure(self.volvo_window, 0, weight=0)
        Grid.columnconfigure(self.volvo_window, 1, weight=1)

    def set_entry_values(self):
        # initialize number of levels
        self.time_res = TIME_RESOLUTION
        self.angle_tolerance = ANGLE_TOLERANCE
        self.cluster_dist = CLUSTER_DISTANCE
        self.min_cluster_size = MIN_CLUSTER_SIZE

    def switch_oppo_state(self):
        self.opposite_state.set(1 - self.opposite_state.get())

    def switch_per_state(self):
        self.perpendicular_state.set(1 - self.perpendicular_state.get())

    def switch_mean_state(self):
        self.mean_state.set(1 - self.mean_state.get())

    def switch_mean_by_dir_state(self):
        self.mean_by_dir.set(1 - self.mean_by_dir.get())

    def build_volvo_ui(self):
        # create titles for entry boxes
        for i, label_text in enumerate(VOLVO_ENTRY_LABELS):
            Label(self.entry_frame, text=label_text + "\t").grid(row=i, column=0)

        # create entry boxes

        # distance tolerance
        self.time_res_entry = Entry(self.entry_frame, width=15)
        self.time_res_entry.insert(0, str(self.time_res))
        self.time_res_entry.grid(row=0, column=1)

        # angle tolerance
        self.angle_tolerance_entry = Entry(self.entry_frame, width=15)
        self.angle_tolerance_entry.insert(0, str(self.angle_tolerance))
        self.angle_tolerance_entry.grid(row=1, column=1)

        # cluster distance
        self.cluster_dist_entry = Entry(self.entry_frame, width=15)
        self.cluster_dist_entry.insert(0, str(self.cluster_dist))
        self.cluster_dist_entry.grid(row=2, column=1)

        # cluster distance
        self.min_cluster_size_entry = Entry(self.entry_frame, width=15)
        self.min_cluster_size_entry.insert(0, str(self.min_cluster_size))
        self.min_cluster_size_entry.grid(row=3, column=1)

        # update volvo parameters
        Button(master=self.buttons_frame, text="Update VOLVO\xa9 Parameters",
               command=self.run_volvo).grid(row=0, column=0, sticky="nsew")

        # save volvo lines
        Button(master=self.buttons_frame, text="Export Lines",
               command=self.export_volvo_lines).grid(row=1, column=0, sticky="nsew")

        # apply volvo scan to the original scan
        Button(master=self.buttons_frame, text="Apply VOLVO\xa9",
               command=self.apply_volvo).grid(row=2, column=0, sticky="nsew")

        # cancel volvo
        Button(master=self.buttons_frame, text="Abort VOLVO\xa9",
               command=self.abort_volvo).grid(row=3, column=0, sticky="nsew")

        # remove opposite points
        opposite_checkbutton = Checkbutton(master=self.checkboxes_frame, text="Remove Opposite Points",
                                           command=self.switch_oppo_state)
        opposite_checkbutton.grid(row=0, column=0, sticky="nsew")
        opposite_checkbutton.invoke()

        # remove perpendicular points
        perpendicular_checkbutton = Checkbutton(master=self.checkboxes_frame, text="Remove Perpendicular Points",
                                                command=self.switch_per_state)
        perpendicular_checkbutton.grid(
            row=1, column=0, sticky="nsew")
        perpendicular_checkbutton.invoke()

        # substract mean checkbutton
        mean_checkbutton = Checkbutton(master=self.checkboxes_frame, text="Subtract Mean from Each Line",
                                       command=self.switch_mean_state)
        mean_checkbutton.grid(
            row=2, column=0, sticky="nsew")
        mean_by_dir_checkbutton = Checkbutton(master=self.checkboxes_frame, text="Subtract Mean by line directions",
                                              command=self.switch_mean_by_dir_state)
        mean_by_dir_checkbutton.grid(
            row=3, column=0, sticky="nsew")

        Label(master=self.settings_frame, text=VOLVO_EXPLANATION).pack(side=BOTTOM)

        # bind to keyboard shortcuts
        self.volvo_window.bind("<Return>", self.run_volvo)
        self.volvo_window.bind("<Control-s>", self.export_volvo_lines)

    def draw_volvo_graph(self):
        self.geo_axes[1].clear()

        for line_ind in self.lines_indices_fixed:
            self.geo_axes[1].scatter(self.volvo_scan.x[line_ind], self.volvo_scan.y[line_ind])

        self.canvas.draw()

    def run_volvo(self, *args):
        self.volvo_scan, self.lines_indices_fixed = separate_lines(scan=self.volvo_scan,
                                                                   cluster_distance=float(
                                                                       self.cluster_dist_entry.get()),
                                                                   min_cluster_size=int(
                                                                       self.min_cluster_size_entry.get()),
                                                                   time_resolution=float(self.time_res_entry.get()),
                                                                   perpendicular_state_flag=bool(
                                                                       self.perpendicular_state.get()),
                                                                   angle_tolerance_value=float(
                                                                       self.angle_tolerance_entry.get()),
                                                                   opposite_state_flag=bool(self.opposite_state.get()),
                                                                   mean_by_dir_flag=bool(self.mean_by_dir.get()),
                                                                   substract_mean_per_line_flag=bool(self.mean_state.get()))

        # update graph
        self.draw_volvo_graph()

    def export_volvo_lines(self, *args):
        df = convert_magscan_to_df(self.main_window.scan)

        df["Line Index"] = np.zeros(len(df)) - 1

        for line_ind, indexes_in_line in enumerate(self.lines_indices_fixed):
            df["Line Index"][indexes_in_line] = line_ind

        df["Line Index"] = df["Line Index"].astype(int)

        original_file_name = Path(self.main_window.scan.file_name).name

        # checks if the name of the file starts with prefix and saves accordingly
        if not original_file_name.startswith(LINES_NAME_PREFIX):
            original_file_name = LINES_NAME_PREFIX + original_file_name

        save_path = Path(asksaveasfilename(initialfile=original_file_name))

        df.to_csv(save_path, sep="\t", index=False)

        showinfo(WINDOW_TITLE, "Saved successfully!")

    def abort_volvo(self):
        self.volvo_window.destroy()

    def apply_volvo(self):
        # update scan
        lines_indices_fixed = np.hstack(self.lines_indices_fixed)
        fixed_volvo_scan = self.volvo_scan[lines_indices_fixed.flatten()]
        self.main_window.set_scan(fixed_volvo_scan)

        # close the window
        self.volvo_window.destroy()
