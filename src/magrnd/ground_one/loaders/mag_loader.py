import logging
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from datetime import time
from pathlib import Path
from tkinter.filedialog import askopenfilenames, askopenfilename
from tkinter.messagebox import showerror

import numpy as np
import pandas as pd
import utm
from tqdm import tqdm

from src.magrnd.ground_one.data_processing.consts import GZ_DIR, EXPECTED_TIMESTAMP_LEN, IA_SAMPLE_RATE_HZ, WINDOW_TITLE, \
    VALIDATED_STR
from src.magrnd.ground_one.loaders.file_identifier import identify_scan
from src.magrnd.ground_one.loaders.ia_file_encoding_fixer import fix_encoding
from src.mag_utils.mag_utils.functional import validator
from src.mag_utils.mag_utils.functional.mag_eili_converter import magmap_file_reformatter
from src.mag_utils.mag_utils.loader import blackwidow, base
from src.mag_utils.mag_utils.loader import main_load
from src.mag_utils.mag_utils.scans import AluminumManScan
from src.mag_utils.mag_utils.scans import magScan, HorizontalScan

warnings.filterwarnings("ignore")


def mag_utils_wrapper_func(path_to_scan_file: Path, scan_type):
    SCAN_TYPE_MAPPING = {"widow": blackwidow,
                         "base": base}

    # create gz dir if does not exist
    if not GZ_DIR.exists():
        GZ_DIR.mkdir()

    # hide logging
    logger = logging.getLogger("mag_utils.mag_utils.mag_utils.mag_utils.functional.logger")
    logger.setLevel(logging.INFO)

    # perform validation of data
    validated_file_path = GZ_DIR / "validated.txt"

    # scan_type_object = base

    validator.validate(file_to_validate_path=str(path_to_scan_file),
                       validated_file_path=str(validated_file_path),
                       log_file_path=str(GZ_DIR / "faulty_lines_log.txt"),
                       file_type=scan_type)

    scan = SCAN_TYPE_MAPPING[scan_type].load(validated_file_path.__str__())

    # delete validated file
    validated_file_path.unlink()
    scan.file_name = str(path_to_scan_file)
    return scan


def auto_format_time(time_str: str):
    if time_str is None:
        return
    # remove 1900-01-01 from old GZ IA files
    if '1900-' in time_str:
        time_str = time_str[11:]
    if ":" not in time_str:
        # determine if time is valid, and if not make it valid
        zeros_string = '0' * (6 - len(time_str[:time_str.rindex('.')]))
        time_str = (zeros_string + time_str)
        ms_s = "" if "." not in time_str else time_str[time_str.index(".") + 1:]

        # extract time data from string
        microsecond = int(
            float(ms_s) * 10 ** (EXPECTED_TIMESTAMP_LEN - len(
                ms_s))) if ms_s else 0  # for taking only first 6 digits after the decimal
        hour, minute, second = time_str[:2], time_str[2:4], time_str[4:time_str.index(".")]
    else:
        hour, minute, second = tuple(time_str.split(":"))
        second, microsecond = tuple(second.split(".")) if "." in second else (second, 0)

    return time(hour=int(hour),
                minute=int(minute),
                second=int(second),
                microsecond=int(microsecond))


def load_gz(path_to_scan_file: Path):
    # save original format time function to undo change later
    original_format_time_function = blackwidow.format_time

    # set to use custom function
    blackwidow.format_time = auto_format_time

    # load scan
    scan = blackwidow.lab_txt_format_loader(str(path_to_scan_file))

    # return format time to original state
    blackwidow.format_time = original_format_time_function

    return scan


def load_widow(path_to_scan_file: Path):
    return mag_utils_wrapper_func(path_to_scan_file, "widow")


def load_base(path_to_scan_file: Path):
    return mag_utils_wrapper_func(path_to_scan_file, "base")


def load_ia_raw(path_to_scan_file: Path, path_to_calib_file=None):  # todo: work with mike

    """
    :param path: str scan file path
    :return: ia matrix of bx,by,height,time
    """

    ## LOAD IA DATA ##

    # read all gps data
    gps_df = pd.read_csv(path_to_scan_file, delimiter=',', skiprows=3, names=list(range(15)), skipfooter=2,
                         error_bad_lines=True)
    gps_df = gps_df.iloc[gps_df[1].dropna().index, :]

    # read all sensor data
    b_df = pd.read_csv(path_to_scan_file, delimiter=' ', skiprows=3, names=list(range(8)), skipfooter=2,
                       error_bad_lines=True)
    b_df = b_df.iloc[b_df[7].dropna().index, :]

    gps_df.iloc[:, 2] = gps_df.iloc[:, 2].interpolate()
    gps_df.iloc[:, 4] = gps_df.iloc[:, 4].interpolate()

    height = gps_df.iloc[:, 9].values

    # handle time format
    time = gps_df.iloc[:, 1].values
    time_s = [t.zfill(8) for t in time.astype(str)]
    t_seconds = np.array([int(t[:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:]) for t in
                          time_s])  # time in seconds and not H:M:S

    # find if there is a day transfer within scan
    diff_t = np.diff(t_seconds)
    if any(diff_t < 0):
        next_day_index, = np.where(diff_t < 0)[0]
        t_seconds[next_day_index + 1:] += 3600 * 24

    gps_index = b_df.iloc[:, 2].values

    vectorized_translate_gps = np.vectorize(blackwidow.translate_gps)
    x_utm, y_utm = vectorized_translate_gps(gps_df.iloc[:, 4].values, gps_df.iloc[:, 2].values)

    sensor_df_new = pd.DataFrame({'bx': b_df.iloc[:, 5],
                                  'by': b_df.iloc[:, 6],
                                  'bz': b_df.iloc[:, 7],
                                  'gps_index': gps_index})
    gps_df_new = pd.DataFrame(
        {'x': x_utm,
         'y': y_utm,
         'height': height,
         'time': t_seconds})

    sensor_df_new = sensor_df_new.append({'x': None, 'y': None, 'height': None, 'time': None},
                                         ignore_index=True)

    # merging gps and b ia data and interpolating data
    groups = sensor_df_new.groupby('gps_index')
    ia_data_matrix = pd.DataFrame()
    index_for_adding_gps = 0

    for group in tqdm(groups):
        if index_for_adding_gps >= len(gps_df_new['x']):
            pass
        else:
            index_for_group = group[1].index[0]
            group[1]['x'][index_for_group] = gps_df_new['x'][index_for_adding_gps]
            group[1]['y'][index_for_group] = gps_df_new['y'][index_for_adding_gps]
            group[1]['height'][index_for_group] = gps_df_new['height'][index_for_adding_gps]
            group[1]['time'][index_for_group] = gps_df_new['time'][index_for_adding_gps]

        ia_data_matrix = pd.concat([ia_data_matrix, group[1]])
        index_for_adding_gps += 1
    ia_data_matrix.drop(columns='gps_index', inplace=True)
    ia_data_matrix = ia_data_matrix.astype('float64').interpolate()

    # translate numbers back to time
    ia_data_matrix["time"] = [
        (datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0) + timedelta(seconds=t)).time() for t in
        ia_data_matrix["time"]]

    ## CALIBRATE
    from src.magrnd.ground_one.data_processing.ia_calibration import apply_ia_calib
    calib_xyz = apply_ia_calib(ia_data_matrix, path_to_calib_file)
    ia_data_matrix["B"] = np.linalg.norm(calib_xyz, axis=1)

    if path_to_calib_file is None:
        return ia_data_matrix

    return magScan(file_name=str(path_to_scan_file),
                   x=ia_data_matrix['x'],
                   y=ia_data_matrix['y'],
                   a=ia_data_matrix['height'],
                   b=ia_data_matrix['B'],
                   time=ia_data_matrix['time'],
                   is_base_removed=True)


def load_ia_calibrated(path_to_scan_file: Path):
    df = pd.read_csv(path_to_scan_file, delimiter=r"\s+")
    vectorized_from_latlon = np.vectorize(utm.from_latlon)
    gps = vectorized_from_latlon(df["Latitude"], df["Longitude"], 36)

    return magScan(file_name=str(path_to_scan_file),
                   x=gps[0],
                   y=gps[1],
                   a=df['Altitude'],
                   b=df['MagneticField'],
                   time=df['Time'],
                   is_base_removed=False)


def universal_loader(path_to_scan_file: Path, path_to_calib_file: Path = None):
    # fix file encoding if needed
    fix_encoding(path_to_scan_file, path_to_calib_file)

    scan_type = identify_scan(path_to_scan_file)
    if scan_type == "widow":
        return load_widow(path_to_scan_file)
    elif scan_type == "base":
        return load_base(path_to_scan_file)
    elif scan_type == "ia_raw":
        # if path to calib file was not passed, ask user to provide it
        if path_to_calib_file is None:
            path_to_calib_file = Path(askopenfilename(title="Select calibration file"))
        scan = load_ia_raw(path_to_scan_file, path_to_calib_file)
        scan.sampling_rate = IA_SAMPLE_RATE_HZ
        return scan
    elif scan_type == "ia_calibrated":
        return load_ia_calibrated(path_to_scan_file)
    elif scan_type == "gz":
        return load_gz(path_to_scan_file)
    else:
        showerror(WINDOW_TITLE, "File type not recognized. Please try again")
        raise NotImplementedError


def load(scan_paths: list = None):
    if scan_paths is None:
        scan_paths = [Path(file_path) for file_path in
                      askopenfilenames(title="Select file",
                                       filetypes=[("All Files", "*.*")])]

    # if one file is selected, make it the scan path, else opt for vertical scan loading
    if isinstance(scan_paths, Path):
        scan_paths = [scan_paths]

    if not len(scan_paths):
        raise ValueError("Invalid number of files selected.")

    loaded_scans = []
    for scan_path in scan_paths:
        # handle mag eili files
        if scan_path.suffix == ".dat":
            scan_path = magmap_file_reformatter(scan_path)

        # try loading thru main_load, if unsuccessful, use universal loader
        try:
            scan = main_load.load(str(scan_path), validate=True)

            if isinstance(scan, AluminumManScan):
                raise Exception("Aluminum man is not supported in mag_utils.mag_utils.mag_utils.mag_utils yet!")

            # removing "validated" from the file name
            if VALIDATED_STR in scan.file_name:
                validated_index = scan.file_name.rindex(VALIDATED_STR)
                scan.file_name = scan.file_name[:validated_index] + scan.file_name[
                                                                    validated_index + len(VALIDATED_STR):]
        except Exception as e:
            print(f"Encountered error: {e}, falling back to universal loader..")

        scan = universal_loader(scan_path)

        loaded_scans.append(scan)

    # check if given more than one HorizontaScan if so, join all scans together. if given single scan return only it.
    if isinstance(loaded_scans[0], HorizontalScan):
        if len(loaded_scans) > 1:
            appended_scans = deepcopy(loaded_scans[0])
            for scan in loaded_scans[1:]:
                appended_scans = appended_scans.append(scan)
            return appended_scans

    if len(loaded_scans) == 1:
        return loaded_scans[0]

    return loaded_scans
