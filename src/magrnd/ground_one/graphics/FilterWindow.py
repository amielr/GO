import tkinter as tk
from tkinter.ttk import Button, Label, Frame, Radiobutton
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
from magrnd.ground_one.data_processing.consts import GUI_THEME, FILTER_TITLE, BG_COLOR, \
    FG_COLOR, SAMPLE_RATE_HZ, DEFAULT_LOW_PASS_FILTER_FREQS, \
    RANGE_SELECTOR_ALPHA, VISUALIZATIONS_ON1_OFF0
from copy import deepcopy
from mag_utils.mag_utils.functional.signal_processing import calculate_fft, lowpass_filter, highpass_filter, bandpass_filter

FILTER_BUTTONS = {"Low Pass": lowpass_filter,
                  "High Pass": highpass_filter,
                  "Band Pass": bandpass_filter}


class FilterWindow:
    def __init__(self, window):
        self.main_window = window

        # override sample rate if needed
        if hasattr(self.main_window.scan, "sampling_rate"):
            self.sample_rate = self.main_window.scan.sampling_rate
        else:
            self.sample_rate = SAMPLE_RATE_HZ

        # store crucial data
        self.original_signal_mean = self.main_window.scan.b.mean()
        self.normalized_signal = self.main_window.scan.b - self.original_signal_mean

        # create fft for signal
        self.fft_freq, self.fft_signal = calculate_fft(self.normalized_signal, self.sample_rate)

        # initialize window
        self.initialize_window()

        # build graphs
        self.build_graphs()

        # build ui
        self.build_ui()

        # draw graphs
        self.draw_graphs()

        # apply initial default filtering
        self.selected_filter_type = "Low Pass"
        self.apply_filter(*DEFAULT_LOW_PASS_FILTER_FREQS)

        ###################### Replacement
        self.update_scan()

        # fix fft display
        self.fix_fft_graph()

    def initialize_window(self):
        self.filter_window = ThemedTk(themebg=True)
        self.filter_window.set_theme(GUI_THEME)
        self.filter_window.wm_title(FILTER_TITLE)

        # settings frame
        self.settings_frame = Frame(self.filter_window)
        self.settings_frame.grid(row=0, column=0, sticky="nsew")

        # graphs frame
        self.graphs_frame = Frame(self.filter_window)
        self.graphs_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # allow window resizing
        tk.Grid.rowconfigure(self.filter_window, 0, weight=1)
        tk.Grid.columnconfigure(self.filter_window, 0, weight=0)
        tk.Grid.columnconfigure(self.filter_window, 1, weight=1)

    def build_graphs(self):
        # build signal and fft graphs
        self.filter_fig, (self.signal_ax, self.fft_ax) = plt.subplots(2, 1)

        # set filter figure title
        self.filter_fig.suptitle(FILTER_TITLE, color=FG_COLOR)

        self.signal_ax.set_title("Original vs. Filtered Signal", fontdict={"color": FG_COLOR})
        self.signal_ax.set_xlabel("Index")
        self.signal_ax.set_ylabel("B [nT]")

        self.fft_ax.set_title("Original vs. Filtered FFT", fontdict={"color": FG_COLOR})
        self.fft_ax.set_xlabel("Frequency [Hz]")
        self.fft_ax.set_ylabel("Amplitude")

        # associate filter figure to canvas
        self.canvas = FigureCanvasTkAgg(self.filter_fig, master=self.graphs_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        # self.canvas.mpl_connect('button_press_event', self.span_selector)

        # init filter toolbar
        self.filter_toolbar = NavigationToolbar2Tk(self.canvas, self.graphs_frame)

        # set theme
        self.canvas.figure.patch.set_facecolor(BG_COLOR)

        from magrnd.ground_one.graphics.MainWindow import set_axes_theme, set_toolbars_theme
        set_axes_theme(self.signal_ax, self.fft_ax)
        set_toolbars_theme(self.filter_toolbar)

    def build_ui(self):
        self.selected_filter_type_var = tk.StringVar()
        self.filter_options_label = Label(master=self.settings_frame, text="Select Filter Type:")
        self.filter_options_label.pack(side=tk.TOP)

        # build filter radio buttons
        self.low_pass_radio_button = Radiobutton(master=self.settings_frame, text="Low Pass", value="Low Pass",
                                                 variable=self.selected_filter_type_var,
                                                 command=lambda: self.set_filter_type("Low Pass"))
        self.low_pass_radio_button.pack(side=tk.TOP)
        self.high_pass_radio_button = Radiobutton(master=self.settings_frame, text="High Pass", value="High Pass",
                                                  variable=self.selected_filter_type_var,
                                                  command=lambda: self.set_filter_type("High Pass"))
        self.high_pass_radio_button.pack(side=tk.TOP)
        self.band_pass_radio_button = Radiobutton(master=self.settings_frame, text="Band Pass", value="Band Pass",
                                                  variable=self.selected_filter_type_var,
                                                  command=lambda: self.set_filter_type("Band Pass"))
        self.band_pass_radio_button.pack(side=tk.TOP)

        # set default filter type
        self.low_pass_radio_button.state(['selected'])

        # build apply button
        self.done_button = Button(master=self.settings_frame, text="Done", command=self.update_scan)
        self.done_button.pack(side=tk.TOP)

    def draw_graphs(self):
        self.signal_ax.plot(self.normalized_signal, c="r", label="Original", linestyle="dashed")
        self.filtered_data_handle = self.signal_ax.plot(self.normalized_signal, c="b", label="Filtered",
                                                        linestyle="dashed")
        self.fft_ax.plot(self.fft_freq, self.fft_signal, c="r", label="Original", linestyle="dashed")
        self.filtered_fft_handle = self.fft_ax.plot(self.fft_freq, self.fft_signal, c="b", label="Filtered",
                                                    linestyle="dashed")

        # show legend
        self.signal_ax.legend(facecolor=FG_COLOR)

        self.canvas.draw()

    def fix_fft_graph(self):
        # set ax attributes
        self.fft_ax.set_xticks(np.linspace(self.fft_freq.min(), self.fft_freq.max(), 11))
        self.fft_ax.set_yticks(np.linspace(self.fft_signal.min(), self.fft_signal.max(), 11))
        self.fft_ax.set_xlim(0, self.sample_rate / 2)

    def apply_filter(self, vmin, vmax):
        # checks if there was a previous selection, and if so, deletes it
        if hasattr(self, "range_span_selector"):
            self.fft_ax.patches.clear()

        # create a current selection indication
        self.current_selection_indicator = Rectangle((vmin, self.fft_ax.get_ylim()[0]), vmax - vmin,
                                                     self.fft_ax.get_ylim()[1],
                                                     facecolor='green', alpha=RANGE_SELECTOR_ALPHA)
        self.fft_ax.add_patch(self.current_selection_indicator)
        self.canvas.draw()

        self.range_span_selector = SpanSelector(ax=self.fft_ax, onselect=self.apply_filter, direction="horizontal",
                                                useblit=True,
                                                rectprops=dict(facecolor='red', alpha=RANGE_SELECTOR_ALPHA))

        filter_function = FILTER_BUTTONS[self.selected_filter_type]
        self.filtered_signal = filter_function(self.normalized_signal, vmin, vmax, sample_rate=self.sample_rate)

        # create fft
        self.fft_freq, self.fft_signal = calculate_fft(self.filtered_signal, self.sample_rate)

        # update figure
        self.filtered_data_handle[0].set_ydata(self.filtered_signal)
        self.filtered_fft_handle[0].set_data(self.fft_freq, self.fft_signal)

    def set_filter_type(self, filter_type):
        self.selected_filter_type = filter_type

    def update_scan(self):
        # update scan in main window and close filter window
        filtered_scan = deepcopy(self.main_window.scan)
        filtered_scan.b = self.filtered_signal + self.original_signal_mean  # add back the mean from the original signal
        self.main_window.set_scan(filtered_scan)
        self.filter_window.destroy()


if __name__ == "__main__":
    tk.mainloop()
