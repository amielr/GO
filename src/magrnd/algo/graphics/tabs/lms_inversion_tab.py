from tkinter.ttk import Notebook
from .generic_tab import Tab
from mag_algorithms.inversion import LMSInversion
from mag_utils.scans import HorizontalScan
from ground_one.graphics.MainWindow import set_axes_theme


class LMSInversionTab(Tab):
    def __init__(self, tab_control: Notebook, scan: HorizontalScan):
        super().__init__(tab_control=tab_control, algorithm=LMSInversion, scan=scan)

        self.build_settings_frame()

        self.build_graphs_frame()

    def plot_result(self, result):
        self.processed_scan_ax.clear()

        set_axes_theme(self.processed_scan_ax)

        self.processed_scan_ax.tricontourf(result["x"], result["y"], result["moments_amplitude"], cmap="jet")

        self.processed_scan_ax.scatter(self.scan.x, self.scan.y, marker='o', c='black',
                                       alpha=0.2, s=0.1)

        self.canvas.draw()
