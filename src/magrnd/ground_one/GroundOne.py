from magrnd.ground_one.graphics.SelectFileWindow import SelectFileWindow
from magrnd.ground_one.graphics.MainWindow import MainWindow
from src.mag_utils.mag_utils.scans import HorizontalScan, BaseScan
from magrnd.ground_one.data_processing.buttons import view_base_station

if __name__ == "__main__":

    select_file_window = SelectFileWindow()

    type_check_scan = select_file_window.scans if not isinstance(select_file_window.scans, list) else \
    select_file_window.scans[0]

    # checking HorizontalScan first because only it can be one object instead of list
    if isinstance(type_check_scan, HorizontalScan):
        MainWindow(select_file_window.scans)

    # if scan type is base
    elif isinstance(type_check_scan, BaseScan):
        window = type("FictiveWindow", (object,), {"base_scan": type_check_scan})
        view_base_station(window, display_scan_details=False)
