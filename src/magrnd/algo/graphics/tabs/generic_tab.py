import json
import time
import tkinter
import numpy as np
from pathlib import Path
from tkinter.filedialog import asksaveasfilename
from tkinter.ttk import Notebook, Frame, Entry, Label, Button
from algo.consts import FIELD_NAME_FONT, PARAMETER_ERROR_TITLE, PARAMETER_ERROR_MSG, BOTH, TB_COLOR, BG_COLOR, END
from mag_utils.mag_utils.scans import HorizontalScan
from tkinter.messagebox import showerror
from tkinter import Grid
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from ground_one.graphics.MainWindow import set_axes_theme
from ground_one.loaders.mag_loader import load
import codecs
from dataclasses import dataclass
from typing import Any
from os import getlogin
from datetime import datetime


@dataclass
class Parameter:
    is_overridden: bool
    name: str
    default_value: Any
    type: Any


class Tab:
    def __init__(self, tab_control: Notebook, algorithm, scan: HorizontalScan):
        self.scan = scan
        self.tab = Frame(master=tab_control)
        self.algorithm = algorithm
        self.result_points_df = None

        if not hasattr(self, "overridden_parameters"):
            self.overridden_parameters = []

        tab_control.add(child=self.tab, text=self.algorithm.__name__)

        # create geographic frame
        self.geo_frame = Frame(self.tab)
        self.geo_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # create and set up settings frame
        self.settings_frame = Frame(self.tab)
        self.settings_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # set levels for tricontourf
        self.levels = 20

    def build_settings_frame(self):
        parameters_count = self.algorithm.__init__.__code__.co_argcount

        # skip self, and get only function parameters
        required_parameters = self.algorithm.__init__.__code__.co_varnames[1:parameters_count + 1]
        default_parameter_values = list(self.algorithm.__init__.__defaults__)
        parameter_types = [type(val) for val in default_parameter_values]

        # handle the case where there are parameters with no default values
        if len(required_parameters) > len(default_parameter_values):
            for _ in range(len(required_parameters) - len(default_parameter_values)):
                default_parameter_values.insert(0, "")
                parameter_types.insert(0, float)

        # create parameters object
        self.parameters = [
            Parameter(name=name, default_value=default_val, type=val_type,
                      is_overridden=name in self.overridden_parameters) for
            name, default_val, val_type in
            zip(required_parameters, default_parameter_values, parameter_types)]

        for iField, parameter in enumerate(self.parameters):
            if parameter.is_overridden:
                continue

            if parameter.default_value is None:
                parameter.to_be_removed = True
                continue

            Label(self.settings_frame, text=parameter.name.replace("_", " ").title() + ":", font=FIELD_NAME_FONT).grid(
                row=iField, column=0)
            parameter.entry = Entry(self.settings_frame, width=15)
            parameter.entry.insert(0, str(parameter.default_value))
            parameter.entry.grid(row=iField, column=1)

        self.parameters = [parameter for parameter in self.parameters if not hasattr(parameter, "to_be_removed")]

        run_button = Button(self.settings_frame, text="Run", command=self.run_algorithm)
        run_button.grid(row=len(self.parameters) + 1, column=0)

        reset_button = Button(self.settings_frame, text="Reset Parameters", command=self.reset_parameters)
        reset_button.grid(row=len(self.parameters) + 2, column=0)

        reset_button = Button(self.settings_frame, text="Export to CSV", command=self.export_to_csv)
        reset_button.grid(row=len(self.parameters) + 3, column=0)

    def build_graphs_frame(self, num_graphs=2):
        self.initialize_window(num_graphs=num_graphs)

        self.plot_scan()

    def plot_scan(self):
        self.original_scan_ax.clear()

        set_axes_theme(self.original_scan_ax)

        self.original_scan_ax.tricontour(self.scan.x, self.scan.y,
                                         self.scan.b,
                                         colors="black", linewidths=0.2,
                                         levels=self.levels)
        self.original_scan_ax.tricontour(self.scan.x, self.scan.y,
                                         self.scan.b,
                                         colors="black", linewidths=0.5,
                                         levels=self.levels // 5)
        self.original_scan_ax.tricontourf(self.scan.x, self.scan.y,
                                          self.scan.b, cmap="jet",
                                          levels=self.levels)

        self.original_scan_ax.autoscale()

        self.canvas.draw()

    def initialize_window(self, num_graphs):
        # create fig to display interpolation
        self.geo_fig = plt.figure()
        axes = self.geo_fig.subplots(1, num_graphs, sharex=True, sharey=True)

        # associate interp fig to canvas
        self.canvas = FigureCanvasTkAgg(self.geo_fig, master=self.geo_frame)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=1)

        # init interp toolbar
        self.geo_toolbar = NavigationToolbar2Tk(self.canvas, self.geo_frame)

        self.geo_toolbar.config(background=TB_COLOR)
        self.geo_toolbar._message_label.config(background=TB_COLOR)
        for button in self.geo_toolbar.winfo_children():
            button.config(background=TB_COLOR)

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for ax in axes:
            ax.set_aspect("equal")
            set_axes_theme(ax)

        self.canvas.figure.patch.set_facecolor(BG_COLOR)

        if num_graphs == 2:
            self.original_scan_ax, self.processed_scan_ax = axes
        elif num_graphs == 1:
            self.original_scan_ax, = axes

        Grid.rowconfigure(self.tab, 0, weight=1)
        Grid.columnconfigure(self.tab, 0, weight=0)
        Grid.columnconfigure(self.tab, 1, weight=1)

    def basic_process_kwargs(self, kwargs: dict):
        for parameter in self.parameters:
            if parameter.is_overridden:
                continue

            if parameter.type is None or kwargs[parameter.name] == "None" or kwargs[parameter.name] == "":
                kwargs[parameter.name] = None
                continue

            if parameter.type is bool:
                if kwargs[parameter.name] in ("True", "False"):
                    kwargs[parameter.name] = (kwargs[parameter.name] == "True")
                    continue

            if parameter.type is dict:
                kwargs[parameter.name] = eval(kwargs[parameter.name])
                continue

            kwargs[parameter.name] = parameter.type(kwargs[parameter.name])

        return kwargs

    def process_kwargs(self, kwargs):
        return self.basic_process_kwargs(kwargs)

    def plot_result(self, result):
        raise NotImplemented

    def get_kwargs(self):
        return {parameter.name: parameter.entry.get() for parameter in
                self.parameters if hasattr(parameter, "entry")}

    def run_algorithm(self):
        start_time = time.time()
        kwargs = self.get_kwargs()

        # type-cast and process parameters
        try:
            kwargs = self.process_kwargs(kwargs)
        except ValueError as e:
            showerror(PARAMETER_ERROR_TITLE, PARAMETER_ERROR_MSG + " " + str(e))
            return

        algo = self.algorithm(**kwargs)
        self.result = algo.run(self.scan)
        end_time = time.time()
        self.time_elapsed = end_time - start_time

        self.plot_result(self.result)

    def reset_parameters(self):
        for parameter in self.parameters:
            if not hasattr(parameter, "entry"):
                continue
            parameter.entry.delete(0, END)
            parameter.entry.insert(0, str(parameter.default_value))

    def save(self):
        save_datetime = datetime.now()
        output_info = {"Metadata": {"Path": self.scan.file_name,
                                    "Algorithm": self.algorithm.__name__,
                                    "Time Elapsed (s)": self.time_elapsed if hasattr(self,
                                                                                     "time_elapsed") else -1,
                                    "Run Datetime": save_datetime.strftime('%Y/%m/%d %H:%M:%S'),
                                    "User": getlogin()},
                       "Parameters": self.get_kwargs()}

        if hasattr(self, "result"):
            output_info["Result"] = Tab.convert_problematic_types(self.result)

        # save results file
        scan_path = Path(self.scan.file_name)
        prefix = "AL" if scan_path.stem.startswith("GZ") else "ALGO"
        file_path = prefix + scan_path.stem + f"_{self.algorithm.__name__}_{save_datetime.strftime('%Y_%m_%d_%H_%M_%S')}.json"

        save_file_path = asksaveasfilename(filetypes=(("JSON files", "*.json"),), initialfile=file_path)

        with codecs.open(save_file_path, mode="wb", encoding="utf-8") as save_file:
            json.dump(output_info, save_file, ensure_ascii=False)
            save_file.close()

    @staticmethod
    def convert_problematic_types(d, new_dict=None):
        if new_dict is None:
            new_dict = {}

        for key, value in d.items():
            if isinstance(value, dict):
                new_dict[key] = Tab.convert_problematic_types(value)
            elif isinstance(value, list):
                new_dict[key] = np.asarray(value)
            elif isinstance(value, np.ndarray):
                if value.ndim == 2:
                    new_dict[key], = value.tolist()
                else:
                    new_dict[key] = value.tolist()

        return new_dict

    def load_parameters_and_results(self, json_dict):
        # set parameters
        for parameter in self.parameters:
            if not hasattr(parameter, "entry"):
                continue
            parameter.entry.delete(0, END)
            value = json_dict["Parameters"][parameter.name] if parameter.name in json_dict["Parameters"] else None

            try:
                parameter.entry.insert(0, value)
            except tkinter.TclError:
                pass

        # set results
        self.plot_result(Tab.convert_problematic_types(json_dict["Result"]))

    def set_scan(self, scan: HorizontalScan):
        self.scan = scan

        self.plot_scan()

        if hasattr(self, "processed_scan_ax"):
            self.processed_scan_ax.clear()
            set_axes_theme(self.processed_scan_ax)

    def find_result_points(self):
        raise NotImplemented

    def export_to_csv(self):
        if self.result_points_df is None:
            return
        scan_path = Path(self.scan.file_name)
        save_datetime = datetime.now()
        prefix = "AL" if scan_path.stem.startswith("GZ") else "ALGO"
        save_path = asksaveasfilename(filetypes=(("CSV files", "*.csv"),),
                                      initialfile=prefix + scan_path.stem + f"_{self.algorithm.__name__}_{save_datetime.strftime('%Y_%m_%d_%H_%M_%S')}.csv")
        self.result_points_df.to_csv(save_path, index=False)
