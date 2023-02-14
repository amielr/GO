from pathlib import Path
from tkinter.filedialog import asksaveasfilename, askopenfilename
from tkinter.messagebox import showinfo, showerror#, awidowesno

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter
from shutil import rmtree

from magrnd.ground_one.data_processing.magscan2df import convert_magscan_to_df
from magrnd.ground_one.data_processing.cutting import cut_field, cut_height, cut_rectangle, cut_lines, cut_part, bisect_scan, \
    remove_lines
from magrnd.ground_one.data_processing.consts import FILE_NAME_PREFIX, CACHE_DIR, WINDOW_TITLE, BG_COLOR, FG_COLOR, \
    SAVE_AS_FILETYPES, \
    FILE_NAME_SUFFIX
from magrnd.ground_one.loaders.mag_loader import load
from datetime import datetime, date, timedelta
from magrnd.ground_one.graphics.GeotiffWindow import GeotiffWindow
from magrnd.ground_one.graphics.FilterWindow import FilterWindow
from magrnd.ground_one.graphics.VolvoWindow import VolvoWindow


# region BUTTON FUNCTIONS
def apply_base_station(window):
    # check if base is already applied
    if hasattr(window.scan, "b_before_substraction"):
        apply_again = awidowesno(WINDOW_TITLE,
                               f"There is a base station already applied on the scan presented, would you like to re-apply it?")
        if not apply_again:
            return

    scan_path = Path(window.scan.file_name)
    base_confirmed = False

    # predetermine base station file if possible
    if "Merhavi" in scan_path.parts:
        merhav_full_name = scan_path.parts[scan_path.parts.index("Merhavi") + 1]
        merhav_prefix = merhav_full_name if " " not in merhav_full_name else "".join(
            [l for l in merhav_full_name if l.isupper()])
        base_station_path = scan_path.with_name(
            "BASE_" + merhav_prefix + "_" + scan_path.stem[-10:] + ".txt")
        if base_station_path.exists():
            base_confirmed = awidowesno(WINDOW_TITLE,
                                      f'We detected a matching base station file for this scan ({base_station_path.name}), would you like to apply it?')

    if not base_confirmed:
        base_station_path = Path(askopenfilename(title="Select base station file"))

    window.base_scan = load(base_station_path)

    # save original scan
    try:
        scan_after_substraction = window.scan.subtract_base(window.base_scan.b,
                                                            window.base_scan.time,
                                                            inplace=False)
        window.set_scan(scan_after_substraction)
    except ValueError as e:
        window.error(str(e))


def save(window, save_path=None, is_interactive=True):
    df = convert_magscan_to_df(window.scan)
    original_file_name = Path(window.scan.file_name).name

    # checks if the name of the file starts with GZ and saves accordingly
    if not original_file_name.startswith(FILE_NAME_PREFIX):
        original_file_name = FILE_NAME_PREFIX + original_file_name

    # add .txt to filename when saving
    if not original_file_name.lower().endswith(FILE_NAME_SUFFIX):
        original_file_name += FILE_NAME_SUFFIX

    if save_path is None:
        save_path = Path(asksaveasfilename(initialfile=original_file_name, filetypes=SAVE_AS_FILETYPES))

    df.to_csv(save_path, sep="\t", index=False)

    if is_interactive:
        showinfo(WINDOW_TITLE, "Saved successfully!")


def undo(window):
    existing_file_nums = [int(file_path.name.replace(".csv", "")) for file_path in CACHE_DIR.glob("*.csv")]

    # load scan from cache
    if len(existing_file_nums) > 1:
        cached_file = CACHE_DIR / f"{max(existing_file_nums)}.csv"

        # save original file name
        old_file_name = window.scan.file_name
        sample_rate_overidden = hasattr(window.scan, "sample_rate")

        if sample_rate_overidden:
            old_sample_rate = window.scan.sample_rate

        # load cached scan
        scan = load(cached_file)

        # make sure file_name stay the original path
        scan.file_name = old_file_name
        if sample_rate_overidden:
            scan.sample_rate = old_sample_rate

        window.set_scan(scan, needs_cache=False)

        cached_file.unlink()
    else:
        showerror(WINDOW_TITLE, "Could not perform undo! No changes have been made to the scan.")


def view_base_station(window, display_scan_details=True):
    if not hasattr(window, "base_scan"):
        showerror(WINDOW_TITLE, "Base was not loaded, please load a base scan and try again.")
        return
    # create fig and axis
    fig = plt.figure()
    ax = fig.gca()

    # plot data
    ax.set_title(f"Base Station Graph\n{Path(window.base_scan.file_name).name}", fontdict={"color": FG_COLOR})

    # solve next day problem by creating a mask that defines which samples were recorded the next day
    time_delta_mask = create_next_day_mask(window.base_scan.time)
    fictive_dates = np.array([datetime.combine(date(year=1970, month=1, day=1) + timedelta(days=d), t) for t, d in
                              zip(window.base_scan.time, time_delta_mask)])
    ax.plot(fictive_dates,
            window.base_scan.b, c='r', label="Base Station")

    if display_scan_details:
        # store scan begin and end times, and find relevant times for scan

        scan_begin = datetime.combine(date(year=1970, month=1, day=1), window.scan.time[0])
        contains_next_day = scan_begin > datetime.combine(date(year=1970, month=1, day=1), window.scan.time[-1])
        scan_end = datetime.combine(date(year=1970, month=1, day=1 if not contains_next_day else 2),
                                    window.scan.time[-1])
        scan_duration_mask = (fictive_dates > scan_begin) & (fictive_dates < scan_end)

        # plot scan time period
        scan_duration = timedelta(seconds=(scan_end - scan_begin).seconds)
        ax.set_xlim(scan_begin - scan_duration, scan_end + scan_duration)

        ax.plot(fictive_dates[scan_duration_mask], window.base_scan.b[scan_duration_mask], c="b",
                label="Relevant Period")

    # set axes theme to dark mode
    from ground_one.graphics.MainWindow import set_axes_theme, set_toolbars_theme
    set_axes_theme(ax)
    set_toolbars_theme(fig.canvas.manager.toolbar)
    fig.patch.set_facecolor(BG_COLOR)

    # handle axes
    ax.set_ylabel("B [nT]")
    ax.set_xlabel("Time (HH:MM:SS)")

    # set time format
    formatter = DateFormatter("%H:%M:%S")
    ax.xaxis.set_major_formatter(formatter)

    # display legend
    ax.legend(facecolor=FG_COLOR)

    # display figure
    if not display_scan_details:
        plt.show()
    else:
        fig.show()


def create_next_day_mask(times):
    mask = np.zeros(times.shape)

    try:
        next_day_index, = np.where(np.diff([t.hour for t in times]) < 0)[0]
    except ValueError:
        return mask

    mask[next_day_index + 1:] = 1
    return mask


def open_scan(window):
    if CACHE_DIR.exists():
        rmtree(CACHE_DIR)

    scan_path = Path(askopenfilename(title="Open scan..."))
    scan = load(scan_path)
    window.set_scan(scan)


def toggle_indicators(window):
    # check if flag exists, if it does, toggle it, else create it and set to false
    if not hasattr(window, "indicators_on"):
        window.indicators_on = False
    else:
        window.indicators_on = not window.indicators_on

    # connect hover event to necessary function as needed
    if not window.indicators_on:
        window.left_canvas.mpl_disconnect(window.left_hover_cid)
        window.right_canvas.mpl_disconnect(window.right_hover_cid)
    else:
        window.left_hover_cid = window.left_canvas.mpl_connect('motion_notify_event', window.activate_indicators)
        window.right_hover_cid = window.right_canvas.mpl_connect('motion_notify_event', window.activate_indicators)


def mz(window):
    showinfo(WINDOW_TITLE, "hey")
    window.root.unbind("<Control-y>")


def append_scan(window):
    other_scan = load()
    concatenated_scan = window.scan.append(other_scan)
    concatenated_scan.file_name = "merged_scan"
    window.set_scan(concatenated_scan)


def keyboard_shortcut_callback(window, event):
    if event.state & 4 <= 0:
        return

    key_pressed = str(chr(event.keycode)).lower()

    if key_pressed in KEYBOARD_SHORTCUTS:
        shortcut_callback_func = KEYBOARD_SHORTCUTS[key_pressed]
        shortcut_callback_func(window)


# endregion


# region BUTTON DEFINITION
KEYBOARD_SHORTCUTS = {
    "z": undo,
    "s": save,
    "o": open_scan,
    "b": apply_base_station,
    "g": GeotiffWindow
}

RIGHT_BUTTONS = {
    "Cut by X": cut_height,
    "Cut Field": cut_field,
    "Cut Out Section": cut_part
}

LEFT_BUTTONS = {
    "Cut Lines": cut_lines,
    "Remove Lines": remove_lines,
    "Cut Rectangle": cut_rectangle,
    "Bisect": bisect_scan
}

CENTER_BUTTONS = {
    "Apply Base Station": apply_base_station,
    "View Base Station...": view_base_station,
    "Append Scan...": append_scan,
    "Filter Data...": FilterWindow,
    "Open Geotiff Window...": GeotiffWindow,
    "VOLVO \xa9": VolvoWindow
}

FILE_MENU_COMMANDS = {
    "Open (Ctrl+o)": open_scan,
    "Save (Ctrl+s)": save
}

EDIT_MENU_COMMANDS = {
    "Undo (Ctrl+z)": undo
}

VIEW_MENU_COMMANDS = {
    "Toggle Indicators": toggle_indicators
}
# endregion
