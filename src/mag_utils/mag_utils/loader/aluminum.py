"""Functions for loading a aluminum man scan."""

import warnings
import io
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from mag_utils.mag_utils.scans.aluminum_man import AluminumManScan
from mag_utils.mag_utils.functional.time import convert_datetime_to_seconds
from mag_utils.mag_utils.loader import blackwidow

from mag_utils.mag_utils.loader.blackwidow import format_time

pd.options.mode.chained_assignment = None

GNGGA_COLUMNS = (
    'identifier',  # should be always &GPGGA
    'time',  # hhmmss.sss in UTC
    'latitude',
    'N/S',
    'longitude',
    'E/W',
    'fix_quality',  # integer between 1-5
    'num_Satellites',  # number of satellites the gps have in its line of sight
    'hdop',  # horizontal dilution of precision
    'altitude',
    'height_earth',  # usually meters, M
    'differential_time',
    'differential_reference',
    'unknown',  # dont know what is
    'xor_check'  # XOR check value.
)

ALUMINUM_COLUMNS = ('CHANNEL',
                    'NUMERO_TRAME',
                    'INDICE_GPS',
                    'FLAG',
                    'TEMPERATURE1',
                    'BX',
                    'BY',
                    'BZ'
                    )


def fix_encoding(path):
    """
    Get the path of the file and rewrite ascii to file, this function is to fix UNIT 51 files.

    Args:
        path: path of file.
    """
    # convert to pathlib.Path
    if isinstance(path, str):
        path = Path(path)

    # open file with regular encoding
    with path.open('r', encoding='utf-8') as file:
        try:
            file_text = file.read()
        except UnicodeDecodeError:
            warnings.warn("Could not read file using utf-8 encoding, falling back to mag_utils.mag_utils.mag_utils.mag_utils reader...")

        # if no problem with encoding, continue
        if not file_text.isascii():
            # apply new encoding
            encoded_text = file_text.encode('ascii', 'ignore').decode()

            with path.open('w') as encoded_file:
                encoded_file.write(encoded_text)


def get_samples_from_file(path):
    """
    Get the aluminum file and return samples.

    Args:
        path: path of file.

    Returns:
        samples data frame.
    """
    # Extract from the file only the lines that start with '>M01' the moment line.
    samples_lines = "".join([line for line in open(path, 'r')
                             if line.startswith('>M01')])
    samples_df = pd.read_csv(io.StringIO(samples_lines), sep=' ', names=ALUMINUM_COLUMNS)

    return samples_df


def get_gps_from_file(path):
    """
    Get the aluminum file and return gps.

    Args:
        path: path of file.

    Returns:
        gps data frame.
    """
    # Extract from the file only the lines that start with '>G01' the gps lines.
    gps_lines = "".join([line for line in open(path, 'r')
                         if line.startswith('>G01')])
    gps_df = pd.read_csv(io.StringIO(gps_lines), sep=',', names=GNGGA_COLUMNS)

    return gps_df


def make_ia_table(samples_df, gps_df):
    """
    Join the samples data frame and gps df to one data frame.

    Args:
        samples_df: samples_df.
        gps_df: gps_df.

    Returns:
        aluminium man data frame.
    """
    groups = samples_df.groupby('gps_index')
    ia_data_matrix = pd.DataFrame()

    for index_group, group in enumerate(groups):
        if index_group >= len(gps_df['x']):
            break

        group_data = group[1]
        index_of_first_value_in_group = group[1].index[0]
        group_data['x'][index_of_first_value_in_group] = gps_df['x'][index_group]
        group_data['y'][index_of_first_value_in_group] = gps_df['y'][index_group]
        group_data['height'][index_of_first_value_in_group] = gps_df['height'][index_group]
        group_data['time'][index_of_first_value_in_group] = gps_df['time'][index_group]

        ia_data_matrix = pd.concat([ia_data_matrix, group_data], ignore_index=True)

    ia_data_matrix.drop(columns='gps_index', inplace=True)

    return ia_data_matrix


def txt_loader(path):
    """
    Load aluminium man txt files.

    Args:
        path: path of files.

    Returns:
        AluminiumMan object with the value of file.
    """
    fix_encoding(path)

    samples_new, gps_df = get_samples_from_file(path), get_gps_from_file(path)
    gps_df["latitude"] = gps_df["latitude"].interpolate()
    gps_df["longitude"] = gps_df["longitude"].interpolate()
    height = gps_df["altitude"].values
    gps_index = samples_new["INDICE_GPS"].values

    # handle to time format and convert to seconds
    time = gps_df["time"].astype(str).apply(format_time).apply(convert_datetime_to_seconds)
    x_utm, y_utm = blackwidow.translate_gps(gps_df["longitude"].values, gps_df["latitude"].values)

    samples_new = pd.DataFrame({'bx': samples_new["BX"],
                                'by': samples_new["BY"],
                                'bz': samples_new["BZ"],
                                'gps_index': gps_index
                                })
    gps_new = pd.DataFrame(
        {'x': x_utm,
         'y': y_utm,
         'height': height,
         'time': time})

    samples_new = samples_new.append({'x': None,
                                      'y': None,
                                      'height': None,
                                      'time': None},
                                     ignore_index=True)  # add None column to samples data frame

    ia_data_matrix = make_ia_table(samples_new, gps_new)
    ia_data_matrix = ia_data_matrix[:-1].astype('float64').interpolate()  # intepolate all the lines exept the last line
    ia_data_matrix["time"] = [
        (datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0) + timedelta(seconds=t)).time() for t in
        ia_data_matrix["time"]]

    return AluminumManScan(file_name=str(path),
                           x=np.array(ia_data_matrix['x'].to_list()),
                           y=np.array(ia_data_matrix['y'].to_list()),
                           a=np.array(ia_data_matrix['height'].to_list()),
                           bx=np.array(ia_data_matrix['bx'].to_list()),
                           by=np.array(ia_data_matrix['by'].to_list()),
                           bz=np.array(ia_data_matrix['bz'].to_list()),
                           time=np.array(ia_data_matrix['time'].to_list()),
                           is_calibrated=False,
                           is_base_removed=True)


def h5_loader(path: str) -> AluminumManScan:
    """
    Load aluminium man scan h5 file into AluminiumMan object.

    Args:
        path: full path of Aluminium Man h5 file.

    Returns:
        magScan object with the data of the file.
    """
    mag_scan_part = blackwidow.load(path)

    with h5py.File(path, 'r') as h5f:
        bx = np.array(h5f.get('bx'))
        by = np.array(h5f.get('by'))
        bz = np.array(h5f.get('bz'))

    # Create AluminiumMan object with the data from the h5 file.
    return AluminumManScan(file_name=mag_scan_part.file_name,
                           x=mag_scan_part.x,
                           y=mag_scan_part.y,
                           a=mag_scan_part.a,
                           bx=bx,
                           by=by,
                           bz=bz,
                           time=mag_scan_part.time,
                           date=mag_scan_part.date,
                           is_base_removed=mag_scan_part.is_base_removed,
                           interpolated_data=mag_scan_part.interpolated_data,
                           )


def load(path: str) -> AluminumManScan:
    """
    Load a aluminium man scan file into AluminiumMan object.

    Supported file types - h5, txt

    Args:
        path: full path of aluminium man scan file.

    Returns:
        AluminiumMan object with the data of the file.
    """
    if path.lower().endswith('.h5') or path.lower().endswith('.hdf5'):
        scan = h5_loader(path)

    elif path.lower().endswith('.txt'):
        scan = txt_loader(path)

    else:
        raise ValueError(f'Unsupported file type.'
                         f' Only supporting h5 files.'
                         f' file: {path}')

    return scan
