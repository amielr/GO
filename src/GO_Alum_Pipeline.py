from magrnd.ground_one.graphics.SelectFileWindow import SelectFileWindow
from magrnd.ground_one.graphics.MainWindow import MainWindow
from src.mag_utils.mag_utils.scans import HorizontalScan, BaseScan
from magrnd.ground_one.data_processing.buttons import view_base_station


## 1) Load file and calibration file
## 2) Select calibration area, implement calibration
## 3) Implement scalarization


def load_file_and_calibration():
    select_file_window = SelectFileWindow()

    print(f"our scans are: {type(select_file_window.scans)}, {select_file_window.scans}")
    print(f"our scans are: {type(select_file_window.scans.a)}, {select_file_window.scans.a}")




if __name__ == "__main__":
    print("Hello world")
    load_file_and_calibration()

