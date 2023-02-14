from ground_one.graphics.SelectFileWindow import SelectFileWindow
from mag_utils.scans import HorizontalScan
from graphics.algui_window import Algui

if __name__ == "__main__":
    select_file_window = SelectFileWindow()

    # if scan type is a regular mag scan
    if isinstance(select_file_window.scans, HorizontalScan):
        Algui(select_file_window.scans)

    # if scan type is other.
    else:
        raise TypeError(f"Scans of type {type(select_file_window.scans)} are not supported.")