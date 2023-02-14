import numpy as np
from magrnd.ground_one.data_processing.consts import FLIP_SELECTION_SHORTCUT


class LineSelector:
    def __init__(self, window, onselect):
        self.active = False
        self.onselect = onselect
        self.window = window
        self.press_cid = self.window.left_canvas.mpl_connect("button_press_event", self.on_press)
        self.release_cid = self.window.left_canvas.mpl_connect("button_release_event", self.on_release)
        self.draw_cid = self.window.left_canvas.mpl_connect("draw_event", self.on_draw)

    def set_active(self, flag):
        if not flag:
            self.window.left_canvas.mpl_disconnect(self.press_cid)
            self.window.left_canvas.mpl_disconnect(self.draw_cid)
        self.active = flag

    def on_press(self, event):
        self.set_active(True)
        mouse_x, mouse_y = event.xdata, event.ydata
        self.start_ind = np.argmin(
            (self.window.scan.x - mouse_x) ** 2 + (self.window.scan.y - mouse_y) ** 2)
        self.hover_cid = self.window.left_canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.window.left_canvas.mpl_disconnect(self.press_cid)

    def on_release(self, event):
        mouse_x, mouse_y = event.xdata, event.ydata
        self.end_ind = np.argmin(
            (self.window.scan.x - mouse_x) ** 2 + (self.window.scan.y - mouse_y) ** 2)
        self.window.left_canvas.mpl_disconnect(self.release_cid)
        self.window.left_canvas.mpl_disconnect(self.hover_cid)
        self.onselect(self.window, slice(min(self.start_ind, self.end_ind),
                                         max(self.start_ind, self.end_ind)))

    # def on_hover(self, event):
    #

    def on_hover(self, event):
        if self.active and event.inaxes:
            mouse_x, mouse_y = event.xdata, event.ydata
            mid_ind = np.argmin(
                (self.window.scan.x - mouse_x) ** 2 + (self.window.scan.y - mouse_y) ** 2)
            min_ind = min(self.start_ind, mid_ind)
            max_ind = max(self.start_ind, mid_ind)
            selection_marks = self.window.left_ax.plot(self.window.scan.x[min_ind:max_ind],
                                                       self.window.scan.y[min_ind:max_ind], marker="o", c="g")

            # restore the background
            self.window.left_canvas.restore_region(self.window.left_bg)

            # draw indicators
            self.window.left_ax.draw_artist(selection_marks[0])

            # BLIT!!! AND BBOX!!!
            self.window.left_canvas.blit(self.window.left_fig.bbox)

    def on_draw(self, event):
        self.left_bg = self.window.left_canvas.copy_from_bbox(self.window.left_fig.bbox)


# This is an interpretation of Zero's cut up/ cut down function but it is way better because it was written,
# tested and made operational by Itamar Gordetwidow the king
class BisectSelector:
    def __init__(self, window, onselect):
        self.active = False
        self.onselect = onselect
        self.window = window
        self.press_cid = self.window.left_canvas.mpl_connect("button_press_event", self.on_press)
        self.release_cid = self.window.left_canvas.mpl_connect("button_release_event", self.on_release)
        self.draw_cid = self.window.left_canvas.mpl_connect("draw_event", self.on_draw)
        self.flip_selection = False
        self.points = np.array([np.array([x, y]) for x, y in zip(self.window.scan.x, self.window.scan.y)])

    def set_active(self, flag):
        if not flag:
            self.window.left_canvas.mpl_disconnect(self.press_cid)
            self.window.left_canvas.mpl_disconnect(self.draw_cid)
        self.active = flag

    def toggle_flip_selection(self, _):
        self.flip_selection = not self.flip_selection

    def on_press(self, event):
        if not event.inaxes:
            return

        self.set_active(True)
        self.window.root.bind(FLIP_SELECTION_SHORTCUT, self.toggle_flip_selection)
        self.initial_x, self.initial_y = event.xdata, event.ydata
        self.hover_cid = self.window.left_canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.window.left_canvas.mpl_disconnect(self.press_cid)

    def on_release(self, event):
        if not self.active or not event.inaxes:
            return

        self.final_x, self.final_y = event.xdata, event.ydata
        self.window.left_canvas.mpl_disconnect(self.release_cid)
        self.window.left_canvas.mpl_disconnect(self.hover_cid)
        least_points_mask = self.create_least_points_mask(self.initial_x, self.initial_y, self.final_x, self.final_y)

        self.onselect(self.window, np.logical_not(least_points_mask))
        self.window.root.unbind(FLIP_SELECTION_SHORTCUT)
        self.flip_selection = False

    def on_hover(self, event):
        if not self.active or not event.inaxes:
            return

        mouse_x, mouse_y = event.xdata, event.ydata
        least_points_mask = self.create_least_points_mask(self.initial_x, self.initial_y, mouse_x, mouse_y)
        self.selection_marks = self.window.left_ax.plot(self.window.scan.x[least_points_mask],
                                                        self.window.scan.y[least_points_mask], marker="o", c="r")

        # restore the background
        self.window.left_canvas.restore_region(self.window.left_bg)

        # draw indicators
        self.window.left_ax.draw_artist(self.selection_marks[0])

        # BLIT!!! AND BBOX!!!
        self.window.left_canvas.blit(self.window.left_fig.bbox)

    def create_least_points_mask(self, initial_x, initial_y, final_x, final_y):
        # calculate line parameters
        m = (final_y - initial_y) / (final_x - initial_x)
        b = initial_y - m * initial_x

        # convert points to vectors
        initial_point, final_point = np.array([initial_x, initial_y]), np.array([final_x, final_y])
        distance = final_point - initial_point

        # calculate projection (compute the points whose projection is within the line segment).
        projection_mask = np.abs(
            np.array([distance / np.sum(distance ** 2)]) @ (
                    self.points - (initial_point + final_point) / 2).T).flatten() <= 0.5

        # check which points are above the line
        up_mask = self.points[:, 1] > m * self.points[:, 0] + b

        # find the points that are both within the projection area and above the selected line
        mask = projection_mask & up_mask

        # ^ is xor operation (exclusive or)
        if not ((np.count_nonzero(mask) < np.count_nonzero(projection_mask) // 2) ^ self.flip_selection):
            mask = projection_mask & np.logical_not(up_mask)

        return mask

    def on_draw(self, event):
        self.left_bg = self.window.left_canvas.copy_from_bbox(self.window.left_fig.bbox)


class LineSelector:
    def __init__(self, window, onselect):
        self.active = False
        self.onselect = onselect
        self.window = window
        self.press_cid = self.window.left_canvas.mpl_connect("button_press_event", self.on_press)
        self.release_cid = self.window.left_canvas.mpl_connect("button_release_event", self.on_release)
        self.draw_cid = self.window.left_canvas.mpl_connect("draw_event", self.on_draw)

    def set_active(self, flag):
        if not flag:
            self.window.left_canvas.mpl_disconnect(self.press_cid)
            self.window.left_canvas.mpl_disconnect(self.draw_cid)
        self.active = flag

    def on_press(self, event):
        self.set_active(True)
        mouse_x, mouse_y = event.xdata, event.ydata
        self.start_ind = np.argmin(
            (self.window.scan.x - mouse_x) ** 2 + (self.window.scan.y - mouse_y) ** 2)
        self.hover_cid = self.window.left_canvas.mpl_connect("motion_notify_event", self.on_hover)
        self.window.left_canvas.mpl_disconnect(self.press_cid)

    def on_release(self, event):
        mouse_x, mouse_y = event.xdata, event.ydata
        self.end_ind = np.argmin(
            (self.window.scan.x - mouse_x) ** 2 + (self.window.scan.y - mouse_y) ** 2)
        self.window.left_canvas.mpl_disconnect(self.release_cid)
        self.window.left_canvas.mpl_disconnect(self.hover_cid)
        self.onselect(self.window, slice(min(self.start_ind, self.end_ind),
                                         max(self.start_ind, self.end_ind)))

    # def on_hover(self, event):
    #

    def on_hover(self, event):
        if self.active and event.inaxes:
            mouse_x, mouse_y = event.xdata, event.ydata
            mid_ind = np.argmin(
                (self.window.scan.x - mouse_x) ** 2 + (self.window.scan.y - mouse_y) ** 2)
            min_ind = min(self.start_ind, mid_ind)
            max_ind = max(self.start_ind, mid_ind)
            selection_marks = self.window.left_ax.plot(self.window.scan.x[min_ind:max_ind],
                                                       self.window.scan.y[min_ind:max_ind], marker="o", c="g")

            # restore the background
            self.window.left_canvas.restore_region(self.window.left_bg)

            # draw indicators
            self.window.left_ax.draw_artist(selection_marks[0])

            # BLIT!!! AND BBOX!!!
            self.window.left_canvas.blit(self.window.left_fig.bbox)

    def on_draw(self, event):
        self.left_bg = self.window.left_canvas.copy_from_bbox(self.window.left_fig.bbox)


class LineRemover(LineSelector):
    def on_release(self, event):
        mouse_x, mouse_y = event.xdata, event.ydata
        self.end_ind = np.argmin(
            (self.window.scan.x - mouse_x) ** 2 + (self.window.scan.y - mouse_y) ** 2)
        self.window.left_canvas.mpl_disconnect(self.release_cid)
        self.window.left_canvas.mpl_disconnect(self.hover_cid)
        mask = np.ones_like(self.window.scan.x)
        mask[min(self.start_ind, self.end_ind): max(self.start_ind, self.end_ind)] = 0
        self.onselect(self.window, mask == 1)

    def on_hover(self, event):
        if self.active and event.inaxes:
            mouse_x, mouse_y = event.xdata, event.ydata
            mid_ind = np.argmin(
                (self.window.scan.x - mouse_x) ** 2 + (self.window.scan.y - mouse_y) ** 2)
            min_ind = min(self.start_ind, mid_ind)
            max_ind = max(self.start_ind, mid_ind)
            selection_marks = self.window.left_ax.plot(self.window.scan.x[min_ind:max_ind],
                                                       self.window.scan.y[min_ind:max_ind], marker="o", c="r")

            # restore the background
            self.window.left_canvas.restore_region(self.window.left_bg)

            # draw indicators
            self.window.left_ax.draw_artist(selection_marks[0])

            # BLIT!!! AND BBOX!!!
            self.window.left_canvas.blit(self.window.left_fig.bbox)

