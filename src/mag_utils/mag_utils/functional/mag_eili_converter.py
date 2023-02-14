from pathlib import Path
import re

import numpy as np
import pandas as pd
import utm

# magEili2widow
#
# A functional script in an attempt to transpose the format of mag Eili raw files to the same format of widow Files,
# Disclaimer: the data that doesn't exist in the original files was forged to fill the format properly to be readable
# and correctly parsed by any future programs
# Therefore parts of it are hard-coded and made for a specific format.

# Lists of every column to remove,add, or partially delete (consider)
COLUMNS_TO_CONSIDER = ['Left_X', 'Left_Y', 'Right_X', 'Right_Y',
                       'left_sensor_X', 'left_sensor_Y', 'right_sensor_X', 'right_sensor_Y', 'GPS_X', 'GPS_X']

COLUMNS_TO_REMOVE = ['DATE', 'MARK', 'LINE', 'DOP', 'QUAL_IND', 'Gradient']
COLUMNS_TO_ADD = ['$GPGGA', 'N', 'E', 2, 11, 0.7, 'M', 18.00, 'M', 24, 'TSTR*65', 1726]


def normalize_column_name(col_name: str):
    """
    Args:
        col_name: a str that contains a name of a column in a magmap .dat file.
    Returns: a lowercase string formatted in our language

    """
    return col_name.lower() \
        .replace("g-858", "b") \
        .replace("sensor", "b") \
        .replace("1", "left") \
        .replace("2", "right") \
        .replace("pos", "gps")


def get_regex_match_idx(regex: re.Pattern, strings: list):
    """
    Args:
        regex: regex pattern to be matched
        strings: a list of strings

    Returns: index of the string that matched the regex pattern

    """
    match_list = list(filter(regex.search, strings))
    if not match_list:
        return None
    match_str = match_list[0]
    return np.where(np.array(strings) == match_str)[0][0]


def extract_column_names_from_file(file):
    """
    Args:
        file: pd.Dataframe of dat file from MagMap.

    Returns: the names of the relevant columns in the specific file.

    """
    normalized_cols = list(map(normalize_column_name, file.columns))

    ordered_regex = (re.compile("(?=.*left)(?=.*x)"),
                     re.compile("(?=.*right)(?=.*x)"),
                     re.compile("(?=.*left)(?=.*y)"),
                     re.compile("(?=.*right)(?=.*y)"),
                     re.compile("(?=.*left)(?=.*b)"),
                     re.compile("(?=.*right)(?=.*b)"),
                     re.compile("(?=.*time)"),
                     re.compile("(?=.*height)"))

    match_idxs = np.array([get_regex_match_idx(pattern, normalized_cols) for pattern in ordered_regex])

    gps_x_index, gps_y_index = get_regex_match_idx(re.compile("(?=.*gps)(?=.*x)"),
                                                   normalized_cols), get_regex_match_idx(re.compile("(?=.*gps)(?=.*y)"),
                                                                                         normalized_cols)
    # if some of the coords are not defined.
    if match_idxs[4] is None and match_idxs[5] is not None:
        match_idxs[1] = gps_x_index
        match_idxs[3] = gps_y_index
    elif match_idxs[5] is None and match_idxs[4] is not None:
        match_idxs[0] = gps_x_index
        match_idxs[2] = gps_y_index

    return tuple(file.columns[match_idxs])


def column_process(file):
    """
    Given the raw data from mag Eili recordings of both the left and the right sensors with
    the following columns:

        ['Pos_1', 'Pos_2', 'left_sensor_X', 'left_sensor_Y', 'right_sensor_X', 'right_sensor_Y',
        'QUAL_IND', 'DOP', 'HEIGHT', 'G-858_1', 'G-858_2', 'LINE', 'MARK', 'TIME', 'DATE'],

    where 'G-858_1', 'G-858_2' are the B values of the left and right sensors (names like Left_x,
    lefty, right_B, right_sensor_b, left_gps, GPS1, GPS2 ... will also work).
    the function creates reorganized data frame that contains the data required
    for the $GPGGA format (x,y,B,h,t) starting with the data of the left sensor and
    then the data of the right sensor.

    If only one sensor is found, the function will assume it's position is in the middle (where the gps is).

    Args:
        file: a pd.DataFrame that contains the data from a MAGMAP dat file.

    Returns: a pd.DataFrame of reorganized file according to GZ format
    """

    # Get the correct names of all the columns (for example left_x instead of left_sensor_X)
    [left_sensor_x_col,
     right_sensor_x_col,
     left_sensor_y_col,
     right_sensor_y_col,
     left_sensor_b_col,
     right_sensor_b_col,
     time_col,
     height_col] = extract_column_names_from_file(file)

    # Set the fields of the reorganized file.
    reorganized_file = pd.DataFrame()

    x_values = [file[col] for col in [left_sensor_x_col, right_sensor_x_col] if col is not None]
    reorganized_file['x'] = np.concatenate(x_values)
    reorganized_file['y'] = np.concatenate(
        [file[col] for col in [left_sensor_y_col, right_sensor_y_col] if col is not None])
    reorganized_file['B'] = np.concatenate(
        [file[col] for col in [left_sensor_b_col, right_sensor_b_col] if col is not None])
    reorganized_file['height'] = np.concatenate([file[height_col] for _ in range(len(x_values))])
    reorganized_file['TIME'] = np.concatenate([file[time_col].str.replace(':', '') for _ in range(len(x_values))])

    # Change B column to start with a " " to support the times bridge convention.
    reorganized_file['B'] = [" " + str(b) for b in reorganized_file['B']]

    for column in COLUMNS_TO_ADD:
        reorganized_file[column] = column

    return reorganized_file


def detranslate_gps(file, x_col_name, y_col_name):
    """
     The sole purpose of utm_to_latlong is to convert from utm to decimal degrees
     and to place thus geographical columns in the correct indices,
     It fulfills it using the module "utm" and a for loop.

     Args:
         file: raw data from mag Eili recordings.
         x_col_name: The X sensor data column name.
         y_col_name: The Y sensor data column name.

     Returns:
         file with updated geographical data (Converted to Decimal Degrees format)
     """

    ulist = []
    for x, y in zip(file[x_col_name], file[y_col_name]):
        utm_coords = utm.to_latlon(x, y, 36, "N")
        ulist.append(utm_coords)

    def detranslate(coord):
        is_positive = coord >= 0
        coord = abs(coord)
        minutes, seconds = divmod(coord * 3600, 60)
        degrees, minutes = divmod(minutes, 60)
        degrees = degrees if is_positive else - degrees
        ans = degrees * 100 + minutes + seconds / 60
        return ans

    for i in ulist:
        yield detranslate(i[0]), detranslate(i[1])


def magmap_file_reformatter(file_path: Path):
    """
    Args:
        file_path: pathlib.Path of a magmap .dat file

    Returns: a pathlib.Path of the new file in the widow format

    """
    if not file_path.suffix == ".dat":
        raise ValueError("File entered is not a .dat file, please try again!")

    a = pd.read_csv(file_path, delim_whitespace=True)
    a = column_process(a)
    lat, lon = [], []
    for x, y in detranslate_gps(a, 'x', 'y'):
        lat.append(x)
        lon.append(y)
    a['lat'], a['lon'] = lat, lon

    # extract only relevant columns
    final_columns = ['$GPGGA', 'TIME', "lat", "N", "lon", 'E', 2, 11, 0.7, 'height', 'M', 18.00, 'M', 24,
                     'TSTR*65', "B", 1726]
    a = a[final_columns]
    a['B'] = a['B'].replace('         ', None)
    a.dropna(inplace=True)

    new_file_path = file_path.parent / (file_path.stem + '_as_widow.txt')
    a.to_csv(new_file_path, index=False, header=False)
    return new_file_path
