import numpy as np
from matplotlib.widgets import SpanSelector, RectangleSelector
from magrnd.ground_one.data_processing.consts import RANGE_SELECTOR_ALPHA, MIN_SELECTION_LENGTH, WINDOW_TITLE
from magrnd.ground_one.data_processing.custom_selectors import LineSelector, BisectSelector, LineRemover
from tkinter.messagebox import showerror


def update_scan(window, indices=None):
    # make sure slices never go out of range
    if isinstance(indices, slice):
        if indices.start < 0:
            indices = slice(0, indices.stop)
        if indices.stop >= len(window.scan):
            indices = slice(indices.start, len(window.scan))
        sample_count = (indices.stop - indices.start) // (indices.step if indices.step is not None else 1)
    else:
        sample_count = np.count_nonzero(indices)

    # check if scan length complies with min scan length
    if sample_count < MIN_SELECTION_LENGTH:
        showerror(WINDOW_TITLE,
                  f"The amount of samples selected is beneath the minimum ({MIN_SELECTION_LENGTH} samples)")

        # reset left axis
        window.left_ax.clear()

        # plot scan on left ax
        window.scan.plot(window.left_ax)

        from magrnd.ground_one.graphics.MainWindow import set_axes_theme
        set_axes_theme(window.left_ax)

        # draw canvases
        window.left_canvas.draw()

        # stop run
        return

    window.set_scan(scan=window.scan[indices])

    # fix order after selection is done
    if isinstance(window.selector, SpanSelector):
        # bring field ax back to front
        window.right_field_ax.set_zorder(
            window.right_height_ax.get_zorder() + 1)

    # hide selector
    window.selector.set_active(False)


def cut_height(window):
    window.right_field_ax.set_zorder(
        window.right_height_ax.get_zorder() + 1)  # work around a matplotlib known bug (issue #10009)
    window.selector = SpanSelector(ax=window.right_field_ax,
                                   onselect=lambda min, max: update_scan(window, slice(int(min),
                                                                                       int(max))),
                                   direction="horizontal", useblit=True,
                                   rectprops=dict(facecolor='green', alpha=RANGE_SELECTOR_ALPHA))


def cut_part(window):
    window.right_field_ax.set_zorder(
        window.right_height_ax.get_zorder() + 1)  # work around a matplotlib known bug (issue #10009)
    window.selector = SpanSelector(ax=window.right_field_ax,
                                   onselect=lambda min, max: update_scan(window,
                                                                         create_cut_part_mask(window, min, max)),
                                   direction="horizontal", useblit=True,
                                   rectprops=dict(facecolor='blue', alpha=RANGE_SELECTOR_ALPHA))


def create_cut_part_mask(window, min, max):
    mask = np.ones(window.scan.x.shape)

    # wrap around index values
    if min < 0:
        min = 0
    if max > len(window.scan.x):
        max = len(window.scan.x)

    mask[int(min):int(max)] = 0
    return mask.astype(dtype=bool)


def cut_field(window):
    window.right_field_ax.set_zorder(
        window.right_height_ax.get_zorder() + 1)  # work around a matplotlib known bug (issue #10009)
    window.selector = SpanSelector(ax=window.right_field_ax,
                                   onselect=lambda min, max: update_scan(window,
                                                                         np.where((window.scan.b > min) & (
                                                                                 window.scan.b < max))),
                                   direction="vertical", useblit=True,
                                   rectprops=dict(facecolor='red', alpha=RANGE_SELECTOR_ALPHA))


def cut_rectangle(window):
    window.selector = RectangleSelector(ax=window.left_ax, onselect=lambda erelease, eclick: update_scan(window,
                                                                                                         np.where((
                                                                                                                          min(erelease.xdata,
                                                                                                                              eclick.xdata) < window.scan.x) &
                                                                                                                  (
                                                                                                                          window.scan.x < max(
                                                                                                                      erelease.xdata,
                                                                                                                      eclick.xdata)) &
                                                                                                                  (
                                                                                                                          min(erelease.ydata,
                                                                                                                              eclick.ydata) < window.scan.y) &
                                                                                                                  (
                                                                                                                          window.scan.y < max(
                                                                                                                      erelease.ydata,
                                                                                                                      eclick.ydata)))),
                                        drawtype='box', useblit=True,
                                        spancoords='pixels', minspanx=5, minspany=5,
                                        interactive=True)


def cut_dot(window):
    from mpl_interactions.widgets import scatter_selector_index
    window.selector = scatter_selector_index(ax=window.left_ax, x=window.scan.x, y=window.scan.y, alpha=0)
    window.selector.on_changed(lambda index: update_scan(window, np.arange(0, len(window.scan.x)) != index))


# deprecated
def cut_lines(window):
    window.selector = LineSelector(window=window, onselect=update_scan)


def remove_lines(window):
    window.selector = LineRemover(window=window, onselect=update_scan)


def bisect_scan(window):
    window.selector = BisectSelector(window=window, onselect=update_scan)
