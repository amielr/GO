import numpy as np
from magrnd.ground_one.data_processing.consts import SARGEL_COLOR
from functools import partial


class Sargel:

    def __init__(self, canvas, ax, x0, y0, toolbar):
        self.sargel_point = [x0, y0]
        self.canvas = canvas
        self.ax = ax
        self.toolbar = toolbar
        self.distance_labels = []
        self.save_bg(None)
        self.canvas.mpl_connect('button_press_event', partial(self.on_press))
        self.canvas.mpl_connect('draw_event', self.save_bg)

    def on_press(self, event):
        if self.toolbar.mode.name == "NONE":
            if event.inaxes == self.ax:
                if hasattr(self, "sargel_line") and len(self.sargel_line):
                    self.sargel_line.pop(0).remove()
                self.sargel_line = self.ax.plot([event.xdata, event.xdata], [event.ydata, event.ydata],
                                                color=SARGEL_COLOR,
                                                linewidth=2)
                self.sargel_point = [event.xdata, event.ydata]

                self.hover_cid = self.canvas.mpl_connect('motion_notify_event', partial(self.on_hover))
                self.release_cid = self.canvas.mpl_connect('button_release_event', partial(self.on_release))

                self.update()

    def update(self):
        # restore the background
        self.canvas.restore_region(self.bg)

        # draw indicators
        if len(self.sargel_line):
            self.ax.draw_artist(self.sargel_line[0])

        if hasattr(self, "distance_label"):
            self.ax.draw_artist(self.distance_label)

        # BLIT!!! AND BBOX!!!
        self.canvas.blit(self.ax.figure.bbox)

    def save_bg(self, _):
        # saves the background of the plots for animated indicators
        self.bg = self.canvas.copy_from_bbox(self.ax.figure.bbox)

    def on_hover(self, event):
        if event.inaxes == self.ax:
            dist = abs(
                np.sqrt((self.sargel_point[0] - event.xdata) ** 2 + (self.sargel_point[1] - event.ydata) ** 2))

            if len(self.sargel_line):
                self.sargel_line[0].set_xdata([self.sargel_point[0], event.xdata])
                self.sargel_line[0].set_ydata([self.sargel_point[1], event.ydata])
                self.distance_label = self.ax.text(s=f'{dist:.4f} m', x=event.xdata, y=event.ydata, color="black",
                                                   backgroundcolor="white")
                self.distance_labels.append(self.distance_label)
                self.update()

    def on_release(self, event):
        if hasattr(self, "sargel_line") and len(self.sargel_line):
            self.sargel_line.pop(0).remove()

        for label in self.distance_labels:
            label.set_visible(False)

        self.canvas.mpl_disconnect(self.hover_cid)
        self.canvas.mpl_disconnect(self.release_cid)
        self.update()
