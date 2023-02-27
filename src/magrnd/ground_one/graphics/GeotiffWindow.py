import tkinter as tk
from pathlib import Path
from tkinter.filedialog import asksaveasfilename
from tkinter.ttk import Button, Checkbutton, Label, Frame, Entry
from ttkthemes import ThemedTk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from magrnd.ground_one.data_processing.consts import GUI_THEME, GEOTIFF_TITLE, AXES_ASPECT, INITIAL_LEVELS_NUM, GZ_DIR, BG_COLOR, \
    TB_COLOR, \
    SCAN_ROUTE_ALPHA
from magrnd.ground_one.graphics.Sargel import Sargel
import matplotlib.pyplot as plt
from mag_utils.mag_utils.functional.geotiff_utils import create_tiff


class GeotiffWindow:
    def __init__(self, window, guiless=False):
        self.main_window = window
        self.guiless = guiless

        # set scan route state for the checkbox
        self.route_state = tk.IntVar()
        self.route_state.set(False)

        # initialize window
        self.initialize_window()

        # set entry values
        self.set_entry_values()

        # build UI - buttons and entries
        self.build_geotiff_ui()

        # draw geotiff
        self.draw_geotiff()

    def set_entry_values(self):
        # initialize number of levels
        self.levels = INITIAL_LEVELS_NUM

        # save min max values
        self.max_val = self.main_window.scan.b.max()
        self.min_val = self.main_window.scan.b.min()

        # contour interval
        self.original_contour_interval = (self.max_val - self.min_val) / self.levels
        self.contour_interval = self.original_contour_interval
        self.contour_interval_entry = self.original_contour_interval

    def initialize_window(self):
        self.geotiff_window = ThemedTk(themebg=True)
        self.geotiff_window.set_theme(GUI_THEME)
        self.geotiff_window.wm_title(GEOTIFF_TITLE)

        # create scan frame
        self.scan_frame = Frame(self.geotiff_window)
        self.scan_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # create and set up settings frame
        self.settings_frame = Frame(self.geotiff_window)
        self.settings_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # create entry frame
        self.entry_frame = Frame(self.settings_frame)
        self.entry_frame.pack(side=tk.TOP)

        # create buttons frame
        self.buttons_frame = Frame(self.settings_frame)
        self.buttons_frame.pack(side=tk.TOP)

        # create histogram frame
        self.hist_frame = Frame(self.settings_frame)
        self.hist_frame.pack(side=tk.TOP)

        # create fig to display interpolation
        self.interp_fig = plt.figure()
        self.interp_ax = self.interp_fig.add_axes([.1, .15, .70, .8])
        self.colorbar_ax = self.interp_fig.add_axes([.85, .15, .05, .8])

        # associate interp fig to canvas
        self.canvas = FigureCanvasTkAgg(self.interp_fig, master=self.scan_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # create histogram and figure and associate to canvas
        self.hist_fig = plt.figure()
        self.hist_ax = self.hist_fig.gca()
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.hist_frame)
        self.hist_canvas.get_tk_widget().grid(row=0, column=0)

        # init interp toolbar
        self.interp_toolbar = NavigationToolbar2Tk(self.canvas, self.scan_frame)

        # set theme and aspect
        self.interp_ax.set_aspect(AXES_ASPECT)

        # creating the Sargel for measuring distance
        self.canvas.mpl_connect("button_press_event", self.on_click)

        for canvas in (self.hist_canvas, self.canvas):
            canvas.figure.patch.set_facecolor(BG_COLOR)

        self.interp_toolbar.config(background=TB_COLOR)
        self.interp_toolbar._message_label.config(background=TB_COLOR)
        for button in self.interp_toolbar.winfo_children():
            button.config(background=TB_COLOR)

        # allow window resizing
        tk.Grid.rowconfigure(self.geotiff_window, 0, weight=1)
        tk.Grid.columnconfigure(self.geotiff_window, 0, weight=0)
        tk.Grid.columnconfigure(self.geotiff_window, 1, weight=1)
        tk.Grid.rowconfigure(self.settings_frame, 0, weight=0)
        tk.Grid.rowconfigure(self.settings_frame, 1, weight=0)
        tk.Grid.rowconfigure(self.settings_frame, 2, weight=0)
        tk.Grid.columnconfigure(self.settings_frame, 0, weight=0)

    def build_geotiff_ui(self):

        # create titles for entry boxes
        for i, label_text in enumerate(("Min Value:", "Max Value:", "Contour Level:")):
            Label(self.entry_frame, text=label_text + "\t").grid(row=i, column=0)

        # create entry boxes
        # min val
        self.min_val_entry = Entry(self.entry_frame, width=15)
        self.min_val_entry.insert(0, str(round(self.min_val, 2)))
        self.min_val_entry.grid(row=0, column=1)

        # max val
        self.max_val_entry = Entry(self.entry_frame, width=15)
        self.max_val_entry.insert(0, str(round(self.max_val, 2)))
        self.max_val_entry.grid(row=1, column=1)

        # contour interval
        self.contour_interval_entry = Entry(self.entry_frame, width=15)
        self.contour_interval_entry.insert(0, str(self.contour_interval))
        self.contour_interval_entry.grid(row=2, column=1)

        # scan route checkbox
        Checkbutton(master=self.buttons_frame, text="Show Scan Route",
                    command=self.switch_route_state).grid(row=0, column=0, sticky="nsew")

        # update scale button
        Button(master=self.buttons_frame, text="Update Scale",
               command=self.update_scale).grid(row=1, column=0, sticky="nsew")

        # reset scale button
        Button(master=self.buttons_frame, text="Reset Scale",
               command=self.reset_scale).grid(row=2, column=0, sticky="nsew")

        # export geotiff
        Button(master=self.buttons_frame, text="Export Geotiff",
               command=self.fix_cb).grid(row=3, column=0, sticky="nsew")

        # bind to keyboard shortcuts
        self.geotiff_window.bind("<Return>", self.update_scale)
        self.geotiff_window.bind("<Control-z>", self.reset_scale)

    def switch_route_state(self):
        self.route_state.set(1 - self.route_state.get())
        self.draw_geotiff()

    def reset_scale(self, *args):
        # set scale back to original scale
        # delete contents
        self.max_val_entry.delete(0, len(self.max_val_entry.get()))
        self.min_val_entry.delete(0, len(self.min_val_entry.get()))
        self.contour_interval_entry.delete(0, len(self.contour_interval_entry.get()))

        # revert back to original values
        self.max_val_entry.insert(0, str(round(self.max_val, 2)))
        self.min_val_entry.insert(0, str(round(self.min_val, 2)))
        self.contour_interval_entry.insert(0, str(self.original_contour_interval))

        # set levels
        self.levels = int((self.max_val - self.min_val) / float(self.contour_interval_entry.get()))

        # redraw geotiff
        self.draw_geotiff()

    def update_scale(self, *args):
        # calculate contour interval
        self.levels = int((self.max_val - self.min_val) / float(self.contour_interval_entry.get()))
        self.contour_interval = (self.max_val - self.min_val) / self.levels

        # delete interval and reset to new one
        self.contour_interval_entry.delete(0, len(self.contour_interval_entry.get()))
        self.contour_interval_entry.insert(0, str(self.contour_interval))

        # redraw geotiff
        self.draw_geotiff()

    def draw_geotiff(self):

        from magrnd.ground_one.graphics.MainWindow import set_axes_theme  # avoid circular import

        self.interp_ax.clear()
        self.hist_ax.clear()
        if hasattr(self, "colorbar_ax"):
            self.colorbar_ax.clear()
            # dark mode
            set_axes_theme(self.hist_ax, self.interp_ax, self.colorbar_ax)
        else:
            # dark mode
            set_axes_theme(self.hist_ax, self.interp_ax)

        # plot interpolated data
        self.interp_ax.tricontour(self.main_window.scan.x, self.main_window.scan.y,
                                  self.main_window.scan.b,
                                  colors="black", linewidths=0.2,
                                  levels=self.levels)
        self.interp_ax.tricontour(self.main_window.scan.x, self.main_window.scan.y,
                                  self.main_window.scan.b,
                                  colors="black", linewidths=0.5,
                                  levels=self.levels // 5)

        # create tiff plot
        self.tiff_plot = self.interp_ax.tricontourf(self.main_window.scan.x, self.main_window.scan.y,
                                                    self.main_window.scan.b, cmap="jet",
                                                    levels=self.levels)
        self.tiff_plot.set_clim(float(self.min_val_entry.get()), float(self.max_val_entry.get()))

        # create colorbar and set limits
        self.cb = self.interp_fig.colorbar(self.tiff_plot, cax=self.colorbar_ax)
        self.cb.vmin = float(self.min_val_entry.get())
        self.cb.vmax = float(self.max_val_entry.get())

        self.hist_ax.hist(self.main_window.scan.b, self.levels)
        self.hist_ax.set_xlim(self.cb.vmin, self.cb.vmax)

        if self.route_state.get() == 1:
            self.interp_ax.scatter(self.main_window.scan.x, self.main_window.scan.y, marker='o', c='black',
                                   alpha=SCAN_ROUTE_ALPHA)

        self.interp_fig.canvas.draw_idle()
        self.hist_fig.canvas.draw_idle()

    def fix_cb(self, *args):
        if self.route_state.get() == 1:
            self.switch_route_state()
        self.cb.remove()
        self.interp_fig.set_frameon(False)
        self.interp_fig.set_dpi(200)
        self.geotiff_window.destroy()
        self.geotiff_export()

    def geotiff_export(self):
        save_path = asksaveasfilename(initialfile=Path(self.main_window.scan.file_name).stem + ".tif")
        create_tiff(self.interp_ax,
                    save_path)

    def on_click(self, event):
        Sargel(self.canvas, self.interp_ax, event.xdata, event.ydata, self.interp_toolbar)
