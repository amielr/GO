from shutil import rmtree
from pathlib import Path

from mag_utils.mag_utils.scans import HorizontalScan
import matplotlib.pyplot as plt
from magrnd.ground_one.data_processing.consts import FG_COLOR, BG_COLOR, AXES_ASPECT, GUI_THEME, WINDOW_TITLE, \
    TB_COLOR, HEIGHT_GRAPH_OFFSET, CACHE_DIR, CACHE_LIMIT, FILE_NAME_FONT, TITLE_FONT, HELP_TEXT
from ttkthemes import ThemedTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter.ttk import LabelFrame, Frame, Button, Label
from tkinter.messagebox import showerror
from magrnd.ground_one.data_processing.buttons import CENTER_BUTTONS, LEFT_BUTTONS, RIGHT_BUTTONS, save, \
    FILE_MENU_COMMANDS, \
    EDIT_MENU_COMMANDS, VIEW_MENU_COMMANDS, keyboard_shortcut_callback
from tkinter import RIGHT, LEFT, TOP, Grid, BOTH, Menu
import numpy as np
from functools import partial

from magrnd.ground_one.graphics.FilterWindow import FilterWindow
from magrnd.ground_one.graphics.VolvoWindow import VolvoWindow


def set_axes_theme(*axes):
    # set theme for each axis
    for ax in axes:
        ax.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        ax.xaxis.label.set_color(FG_COLOR)
        ax.yaxis.label.set_color(FG_COLOR)
        ax.tick_params(colors=FG_COLOR)

        # recolor spines
        ax.spines["left"].set_color(FG_COLOR)
        ax.spines["right"].set_color(FG_COLOR)
        ax.spines["bottom"].set_color(FG_COLOR)
        ax.spines["top"].set_color(FG_COLOR)


def set_toolbars_theme(*toolbars):
    # set color for toolbars and buttons
    for toolbar in toolbars:
        toolbar.config(background=TB_COLOR)
        toolbar._message_label.config(background=TB_COLOR)
        for button in toolbar.winfo_children():
            button.config(background=TB_COLOR)
        toolbar.update()


class MainWindow:


    def __init__(self, scan: HorizontalScan, guiless=False):
        """
    :param
    scan: HorizontalScan: A HorizontalScan object (the scan after load)
    guiless = when True, opens geotiff window without gui. for mor info visit playground.py

        """

        self.scan = scan
        self.guiless = guiless

        # delete cache if already exists
        if CACHE_DIR.exists():
            rmtree(CACHE_DIR)

        if not self.guiless:
            # init window and include graphs in it
            self.initialize_window()

            # cache scan
            self.cache()

            # draw graphs
            self.draw_graphs()

            # create legend
            self.create_interactive_legend()

            # add buttons
            self.construct_ui()
            ############################################
            if not guiless:
                FilterWindow(self)
                VolvoWindow(self).apply_volvo()
            # display window

            self.root.mainloop()

    def draw_graphs(self):
        # plot height on right axis
        self.right_height_ax.plot(self.scan.a, c='g', label='Height')
        self.right_height_ax.set_ylabel("Height [m]")
        self.right_height_ax.set_xlabel("Index")
        self.right_height_ax.set_ylim(self.scan.a.min() - HEIGHT_GRAPH_OFFSET,
                                      self.scan.a.max() + HEIGHT_GRAPH_OFFSET)

        # plot field on right twin axis
        self.right_field_ax.plot(self.scan.b, c='r', label='Field')
        self.right_field_ax.set_ylabel("B [nT]")

        # plot scan on left ax
        self.scan.plot(self.left_ax)

    def create_interactive_legend(self):
        # create and design legend
        right_fig_legend = self.right_fig.legend(loc="upper left")
        right_fig_legend.get_frame().set_alpha(0.4)

        # define pick behavior for legend entries
        for legend_line in right_fig_legend.get_lines():
            legend_line.set_picker(True)
            legend_line.set_pickradius(5)  # 5 pts tolerance

        # set callback for legend picks
        self.right_canvas.mpl_connect('pick_event', self.toggle_right_fig_display)

    def toggle_right_fig_display(self, event):
        # on the pick event, find the plot line corresponding to the
        # legend line, and toggle the visibility
        for axis in self.right_fig.axes:
            for line in axis.lines:
                if event.artist.get_label() == line.get_label():
                    line.set_visible(not line.get_visible())

        # Toggle the visibility on the line in the legend so we can see what lines
        # have been toggled
        event.artist.set_visible(not event.artist.get_visible())
        self.right_canvas.draw()

    def initialize_window(self):
        self.build_graphs()

        # disable ctrl+s shortcut for saving fig, to avoid the dialog opening when saving the scan
        plt.rcParams['keymap.save'].remove("ctrl+s")

        # create window
        self.root = ThemedTk(themebg=True)
        self.root.set_theme(GUI_THEME)
        self.root.wm_title(WINDOW_TITLE)

        # create frames
        self.left_frame = LabelFrame(self.root, text="Location")
        self.right_frame = LabelFrame(self.root, text="Height & Field Graphs")
        self.left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.right_frame.grid(row=0, column=50, padx=5, pady=5, sticky="nsew")

        # create canvas
        self.left_canvas = FigureCanvasTkAgg(self.left_fig, master=self.left_frame)
        self.right_canvas = FigureCanvasTkAgg(self.right_fig, master=self.right_frame)
        for canvas in (self.left_canvas, self.right_canvas):
            canvas.figure.patch.set_facecolor(BG_COLOR)

        # create toolbars
        self.left_toolbar, self.right_toolbar = NavigationToolbar2Tk(self.left_canvas,
                                                                     self.left_frame), NavigationToolbar2Tk(
            self.right_canvas, self.right_frame)

        # set color for toolbars and buttons
        set_toolbars_theme(self.left_toolbar, self.right_toolbar)

        # connect the mouse hover to the indicators
        self.left_hover_cid = self.left_canvas.mpl_connect('motion_notify_event', self.activate_indicators)
        self.right_hover_cid = self.right_canvas.mpl_connect('motion_notify_event', self.activate_indicators)
        self.left_canvas.mpl_connect('draw_event', self.save_bg)
        self.right_canvas.mpl_connect('draw_event', self.save_bg)

        # display graphs on canvases
        self.left_canvas.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=1)
        self.right_canvas.get_tk_widget().pack(side=RIGHT, fill=BOTH, expand=1)


    def save_bg(self, _):
        # saves the background of the plots for animated indicators
        self.left_bg = self.left_canvas.copy_from_bbox(self.left_fig.bbox)
        self.right_bg = self.right_canvas.copy_from_bbox(self.left_fig.bbox)

    def activate_indicators(self, event):
        # KAMPF DIDN'T DO DOCUMENTATION SO WE WILL NOT DO HIS WORK. SHAME ON YOU KAMPF
        if self.left_toolbar.mode.name == "NONE" and self.right_toolbar.mode.name == "NONE":
            mouse_x, mouse_y = event.xdata, event.ydata
            if event.inaxes == self.left_ax:
                ind = np.argmin((self.scan.x - mouse_x) ** 2 + (self.scan.y - mouse_y) ** 2)
                if not self.height_indicator.get_visible():
                    self.height_indicator.set_visible(True)
                self.height_indicator.set_data([ind, ind], [0, 1])
                if not self.location_indicator[0].get_visible():
                    self.location_indicator[0].set_visible(True)
                self.location_indicator[0].set_data(self.scan.x[ind], self.scan.y[ind])
            elif event.inaxes == self.right_field_ax and mouse_x is not None:
                ind = int(round(mouse_x))
                if ind < 0:
                    ind = 0
                elif ind >= len(self.scan.x):
                    ind = len(self.scan.x) - 1
                if not self.location_indicator[0].get_visible():
                    self.location_indicator[0].set_visible(True)
                    self.height_indicator.set_visible(True)
                self.height_indicator.set_data([ind, ind], [0, 1])
                self.location_indicator[0].set_data(self.scan.x[ind], self.scan.y[ind])
            else:
                self.height_indicator.set_visible(False)
                self.location_indicator[0].set_visible(False)

            # restore the background
            self.left_canvas.restore_region(self.left_bg)
            self.right_canvas.restore_region(self.right_bg)

            # draw indicators
            self.left_ax.draw_artist(self.location_indicator[0])
            self.right_field_ax.draw_artist(self.height_indicator)

            # BLIT!!! AND BBOX!!!
            self.left_canvas.blit(self.left_fig.bbox)
            self.right_canvas.blit(self.right_fig.bbox)

    def construct_ui(self):
        # create frames for buttons
        self.left_buttons_frame, self.center_buttons_frame, self.right_buttons_frame = Frame(self.root), Frame(
            self.root), Frame(
            self.root)

        # add title
        Label(self.center_buttons_frame, text=WINDOW_TITLE, font=TITLE_FONT).pack(side=TOP)

        # add filename label
        Label(self.center_buttons_frame, text=Path(self.scan.file_name).stem + "\n", font=FILE_NAME_FONT).pack(side=TOP)

        # set grid locations
        self.center_buttons_frame.grid(row=0, column=25)
        self.left_buttons_frame.grid(row=50, column=0)
        self.right_buttons_frame.grid(row=50, column=50)

        # create aligned tuples to create buttons
        self.button_frames, self.button_defs = (self.left_buttons_frame, self.right_buttons_frame), (
            LEFT_BUTTONS, RIGHT_BUTTONS)

        # create left and right buttons
        for frame, button_def in zip(self.button_frames, self.button_defs):
            for name, func in button_def.items():
                Button(master=frame, command=partial(func, self), text=name).pack(side=LEFT)

        # create center buttons
        for name, func in CENTER_BUTTONS.items():
            Button(master=self.center_buttons_frame, command=partial(func, self), text=name).pack(side=TOP)

        # configure grid locations
        Grid.rowconfigure(self.root, 0, weight=1)
        Grid.columnconfigure(self.root, 0, weight=1)
        Grid.columnconfigure(self.root, 25, weight=1)
        Grid.columnconfigure(self.root, 50, weight=1)
        Grid.rowconfigure(self.center_buttons_frame, 0, weight=1)

        # set up center buttons column
        for i in range(len(CENTER_BUTTONS)):
            Grid.columnconfigure(self.center_buttons_frame, i, weight=1)

        # bind keyboard shortcuts to specific button actions
        self.root.bind("<Key>", lambda event: keyboard_shortcut_callback(window=self, event=event))

        # create main menu
        main_menu = Menu(self.root)
        self.root.config(menu=main_menu)

        # create file menu
        file_menu, edit_menu, view_menu, buttons_menu, help_menu = Menu(main_menu), Menu(main_menu), Menu(
            main_menu), Menu(main_menu), Menu(main_menu)

        # add file and edit menus
        main_menu.add_cascade(label="File", menu=file_menu)
        main_menu.add_cascade(label="Edit", menu=edit_menu)
        main_menu.add_cascade(label="View", menu=view_menu)
        main_menu.add_cascade(label="Buttons", menu=buttons_menu)
        main_menu.add_cascade(label="Help", menu=help_menu)

        for name, func in FILE_MENU_COMMANDS.items():
            file_menu.add_cascade(label=name, command=partial(func, self))

        for name, func in EDIT_MENU_COMMANDS.items():
            edit_menu.add_cascade(label=name, command=partial(func, self))

        for name, func in VIEW_MENU_COMMANDS.items():
            view_menu.add_cascade(label=name, command=partial(func, self))

        for name, func in CENTER_BUTTONS.items():
            buttons_menu.add_cascade(label=name, command=partial(func, self))

        for help_str in HELP_TEXT:
            help_menu.add_cascade(
                label=help_str)

################################

    def build_graphs(self):
        # save figs to attributes
        self.left_fig = plt.figure()  # the figure of the location and the magnetic field values
        self.right_fig = plt.figure()  # the figure of the location and the magnetic field values

        if self.guiless:
            plt.close('all')

        # save axes to attributes
        self.left_ax = self.left_fig.gca()
        self.right_height_ax = self.right_fig.gca()

        # set axis aspect for left axis
        self.left_ax.set_aspect(AXES_ASPECT)

        # create twin x axis for field display
        self.right_field_ax = self.right_height_ax.twinx()

        # set axes theme to create dark mode
        set_axes_theme(self.right_height_ax, self.right_field_ax, self.left_ax)

        # create the height and location indicators
        self.height_indicator = self.right_height_ax.axvline(x=0, color='y', linewidth=2, ls="--", animated=True)
        self.location_indicator = self.left_ax.plot(self.scan.x[0], self.scan.y[0], marker='o', c='black', mfc='none',
                                                    markersize=7, animated=True)

    def set_scan(self, scan: HorizontalScan, needs_cache=True):
        # save to ad-hoc cache
        if needs_cache:
            self.cache()

        # pass sample rate on
        if hasattr(self.scan, "sample_rate"):
            scan.sample_rate = self.scan.sample_rate

        # set scan
        self.scan = scan

        # clear all axes
        self.right_height_ax.clear()
        self.right_field_ax.clear()
        self.left_ax.clear()

        # destroy button frames
        self.right_buttons_frame.destroy()
        self.center_buttons_frame.destroy()
        self.left_buttons_frame.destroy()

        # fix y axis scale change after cut bug
        self.right_field_ax.yaxis.tick_right()
        self.right_height_ax.yaxis.tick_left()

        # redraw graphs
        self.draw_graphs()

        # reconstruct buttons
        self.construct_ui()

        # set theme
        set_axes_theme(self.right_height_ax, self.right_field_ax, self.left_ax)

        # draw canvases
        self.right_canvas.draw()
        self.left_canvas.draw()

    def cache(self):
        # delete cache on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # if cache dir does not exist, create it
        if not CACHE_DIR.exists():
            CACHE_DIR.mkdir(exist_ok=True, parents=True)

        existing_file_nums = [int(file_path.name.replace(".csv", "")) for file_path in CACHE_DIR.glob("*.csv")]

        if len(existing_file_nums) == CACHE_LIMIT:
            oldest_cache_file = CACHE_DIR / f"{min(existing_file_nums)}.csv"
            oldest_cache_file.unlink(missing_ok=True)

        # determine current file number
        file_num = 0 if not len(existing_file_nums) else max(existing_file_nums) + 1

        # save as next file number
        save(self, CACHE_DIR / f"{file_num}.csv", is_interactive=False)


    def on_close(self):
        if CACHE_DIR.exists():
            rmtree(CACHE_DIR)

        self.root.quit()
        self.root.destroy()

    def error(self, msg_content):
        showerror(WINDOW_TITLE, msg_content)
