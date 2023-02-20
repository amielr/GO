import logging
from tkinter.ttk import Notebook

import numpy as np
from matplotlib.widgets import RectangleSelector

from .generic_tab import Tab
from mag_utils.mag_utils.scans import HorizontalScan
from mag_algorithms.pso.pso import ParticleSwarmOptimization
from mag_algorithms.loss.l2_loss import L2Loss
from pso.pso_gui import get_bounds_by_coords
from ground_one.graphics.MainWindow import set_axes_theme
from logging import getLogger, ERROR
import pandas as pd


class Pso(Tab):
    def __init__(self, tab_control: Notebook, scan: HorizontalScan):
        # setup hidden fields
        self.overridden_parameters = ["bounds", "loss", "verbose"]

        super().__init__(tab_control=tab_control,
                         algorithm=ParticleSwarmOptimization,
                         scan=scan)

        logger = getLogger("pyswarms.single.global_best")
        logger.setLevel(ERROR)

        self.build_settings_frame()

        self.build_graphs_frame(num_graphs=1)

    def plot_result(self, result):
        self.original_scan_ax.clear()

        self.plot_scan()

        self.result_points_df = self.find_result_points()

        self.original_scan_ax.scatter(result["x"], result["y"], c="r", edgecolor="black")
        for x, y, d2s, measurement in zip(result["x"], result["y"], result["d2s"], result["measurement"]):
            text_box = self.original_scan_ax.text(x + 1, y + 1,
                                                  f"d2s={round(d2s, 3)} m\nm={round(np.linalg.norm(measurement), 3)} Am^2",
                                                  c="black")
            text_box.set_bbox(dict(facecolor="white", alpha=0.5, linewidth=0))

        # set limits
        bounds = self.get_bounds(n_sources=len(result["x"]))
        self.original_scan_ax.set_xlim(bounds[0][0], bounds[1][0])
        self.original_scan_ax.set_ylim(bounds[0][1], bounds[1][1])

        self.canvas.draw()

    def process_kwargs(self, kwargs: dict):
        kwargs = self.basic_process_kwargs(kwargs)

        # override parameters
        kwargs["bounds"] = self.get_bounds(kwargs['n_sources'])
        kwargs["loss"] = L2Loss(reduction='none')
        kwargs["verbose"] = True

        return kwargs

    def get_bounds(self, n_sources):
        bounds = get_bounds_by_coords(self.scan,
                                      self.rs.geometry[1, :],
                                      self.rs.geometry[0, :],
                                      n_sources=n_sources)
        return bounds

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

        # init rectangle selector
        self.rs = RectangleSelector(self.original_scan_ax, onselect=lambda e1, e2: None, drawtype='box',
                                    useblit=True, button=[1],
                                    minspanx=5, minspany=5, spancoords='pixels', interactive=True)

        self.canvas.draw()

    def find_result_points(self):
        return pd.DataFrame({"x": self.result['x'],
                             "y": self.result['y'],
                             "d2s": self.result['d2s']})
