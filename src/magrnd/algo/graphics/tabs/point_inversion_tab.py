from tkinter.ttk import Notebook

import pandas as pd

from .generic_tab import Tab
from mag_algorithms.inversion import PointFinderInversion
from mag_utils.mag_utils.scans import HorizontalScan
from ground_one.graphics.MainWindow import set_axes_theme
from skimage.feature import peak_local_max


class PointInversion(Tab):
    def __init__(self, tab_control: Notebook, scan: HorizontalScan):
        super().__init__(tab_control=tab_control,
                         algorithm=PointFinderInversion,
                         scan=scan)

        self.build_settings_frame()

        self.build_graphs_frame()

    def plot_result(self, result):
        self.processed_scan_ax.clear()

        set_axes_theme(self.processed_scan_ax)

        self.result_points_df = self.find_result_points()

        self.processed_scan_ax.tricontourf(result["x"], result["y"], result["moments_amplitude"])

        self.processed_scan_ax.scatter(self.scan.x, self.scan.y, marker='o', c='black',
                                       alpha=0.2, s=0.1)

        self.processed_scan_ax.scatter(self.result_points_df['x'], self.result_points_df['y'], c='red')

        self.canvas.draw()

    def find_result_points(self):
        result_matrix = self.result["moments_amplitude"].reshape(self.result["dipoles_rectangle_shape"])
        peak_mask = peak_local_max(result_matrix, indices=False, threshold_rel=0.2)

        return pd.DataFrame({"x": self.result["x"][peak_mask.ravel()],
                             "y": self.result["y"][peak_mask.ravel()],
                             "z": self.result["z"][peak_mask.ravel()]})

