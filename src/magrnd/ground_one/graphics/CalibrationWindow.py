import pandas as pd

from magrnd.ground_one.loaders.mag_loader import load_ia_raw
import matplotlib.pyplot as plt
from threading import Event
from matplotlib.widgets import Button
from magrnd.ground_one.data_processing.consts import APPLY_BUTTON_POS, BG_COLOR, FG_COLOR

from magrnd.ground_one.graphics.MainWindow import set_axes_theme
from magrnd.ground_one.graphics.Calibration_cut import start_calibration_search



class CalibrationWindow:
    def __init__(self, path_to_calibration_file):
        self.indexes_event = Event()
        # self.cut_calibration_part_from_scan(path_to_calibration_file)
        self.cut_callibration_method2(path_to_calibration_file)

    def cut_calibration_part_from_scan(self, path):
        # load IA raw data of a calibration scan
        # cut only the part of the calibration
        # returns GZ_ia_matrix of the selected part

        # close previously opened figures to avoid confusion
        plt.close("all")

        # load ia data
        self.GZ_matrix = load_ia_raw(path)

        # prepare for plotting
        bx, by, bz = self.GZ_matrix[['bx', 'by', 'bz']].to_numpy().T
        self.fig = plt.figure(facecolor=BG_COLOR)
        self.axes = (self.fig.add_subplot(4, 1, 1),
                     self.fig.add_subplot(2, 1, 2))

        # save data length
        self.data_len = len(bx)

        # apply dark mode
        set_axes_theme(*self.axes)

        # plot magnetic data from calib file
        for ax in self.axes:
            ax.plot(bx, linewidth=2, color="r")
            ax.plot(by, linewidth=2, color="g")
            ax.plot(bz, linewidth=2, color="b")
            ax.set_xlabel('Index', fontsize=16)
            ax.set_ylabel('Sensor Value [V]', fontsize=16)
            ax.grid()

        # set titles
        self.axes[1].set_title('Zoom to the part of the calibration', fontsize=22, color=FG_COLOR)
        self.axes[0].set_title('Full Scan Perspective', fontsize=16, color=FG_COLOR)

        # set callbacks
        self.axes[1].callbacks.connect("xlim_changed", self.on_zoom)
        self.axes[1].callbacks.connect("ylim_changed", self.on_zoom)

        # add crop button
        self.crop_button_ax = plt.axes(APPLY_BUTTON_POS)
        self.crop_button = Button(self.crop_button_ax, "Crop")
        self.crop_button.on_clicked(lambda event: self.crop_scan())

        plt.show()

    def cut_callibration_method2(self, path):
        # load IA raw data of a calibration scan
        # cut only the part of the calibration
        # returns GZ_ia_matrix of the selected part

        # close previously opened figures to avoid confusion
        plt.close("all")

        # load ia data
        self.GZ_matrix = load_ia_raw(path)

        # prepare for plotting
        bx, by, bz = self.GZ_matrix[['bx', 'by', 'bz']].to_numpy().T


        print(f'data matrix before cal cut: {self.GZ_matrix.to_numpy().T}')
        self.cut_data, self.cropindexes = start_calibration_search(self.GZ_matrix[['bx', 'by', 'bz']].to_numpy().T)
        print(f"our crop indexes are: {self.cropindexes}")
        bx, by, bz = self.cut_data



        self.fig = plt.figure(facecolor=BG_COLOR)
        self.axes = (self.fig.add_subplot(4, 1, 1),
                     self.fig.add_subplot(2, 1, 2))

        # save data length
        self.data_len = len(bx)

        # apply dark mode
        set_axes_theme(*self.axes)

        # plot magnetic data from calib file
        for ax in self.axes:
            ax.plot(bx, linewidth=2, color="r")
            ax.plot(by, linewidth=2, color="g")
            ax.plot(bz, linewidth=2, color="b")
            ax.set_xlabel('Index', fontsize=16)
            ax.set_ylabel('Sensor Value [V]', fontsize=16)
            ax.grid()

        # set titles
        self.axes[1].set_title('Zoom to the part of the calibration', fontsize=22, color=FG_COLOR)
        self.axes[0].set_title('Full Scan Perspective', fontsize=16, color=FG_COLOR)

        # set callbacks
        self.axes[1].callbacks.connect("xlim_changed", self.on_zoom)
        self.axes[1].callbacks.connect("ylim_changed", self.on_zoom)

        # add crop button
        self.crop_button_ax = plt.axes(APPLY_BUTTON_POS)
        self.crop_button = Button(self.crop_button_ax, "Crop")
        self.crop_button.on_clicked(lambda event: self.crop_scan())

        plt.show()
        return

    def crop_scan(self):
        self.axes[1].set_title('Calculating Calibration...', fontsize=22)
        self.fig.show()

        if not self.cropindexes:
            self.cropped_data = self.GZ_matrix[slice(*map(int, self.axes[1].get_xlim()))].reset_index()
            print("we made it here")
        else:
            print("no we made it here")
            self.cropped_data = self.GZ_matrix.iloc[self.cropindexes, :]
        #self.cropped_data = pd.DataFrame(self.cut_data, columns = ['bx', 'by', 'bz'])

        self.indexes_event.set()

        plt.close(self.fig)

    def get_cropped_data(self):
        while not self.indexes_event.is_set():
            plt.pause(1)

        return self.cropped_data

    def on_zoom(self, ax):
        x0, x1 = ax.get_xlim()

        # remove previously drawn lines
        if hasattr(self, "x0_line"):
            self.x0_line.remove()
        if hasattr(self, "x1_line"):
            self.x1_line.remove()

        self.x0_line, self.x1_line = self.axes[0].axvline(x0, color="y", linewidth=5), self.axes[0].axvline(x1,
                                                                                                            color="y",
                                                                                                            linewidth=5)
