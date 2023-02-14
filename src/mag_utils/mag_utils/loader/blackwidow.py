"""All the functions needed to load blackwidow files."""
import datetime
import h5py
import utm

import pandas as pd
import numpy as np

from mag_utils.mag_utils.scans.horizontal_scan import HorizontalScan
from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan
from mag_utils.mag_utils.loader.base import format_time as base_format_time
from mag_utils.mag_utils.functional.time import format_time

GPGGA_COLUMNS = ('identifier',  # should be always &GPGGA
                 'time',  # hhmmss.sss in UTC
                 'latitude',
                 'N/S',
                 'longitude',
                 'E/W',
                 'fix_quality',  # integer between 1-5
                 'num_Satellites',  # number of satellites the gps have in its line of sight
                 'hdop',  # horizontal dilution of precision
                 'altitude',
                 'altitude_units',  # usually meters, M
                 'geoid_separation',
                 'separation_units',  # usually meters, M
                 'age',  # time in seconds since last DGPA update
                 # usually here goes another field, the DGPA reference station ID
                 'checksum',
                 # now followes two lines that are not part of the gpgga massage
                 'magnetic_field',  # in nanoTesla [nT]
                 'signal_strength'
                 )


def is_gz_file(scan_path: str):
    """
    check if the file is a gz file

    Args:
        scan_path: scan path.

    Returns:
        true if is a gz file.
    """
    with open(scan_path, 'r', errors='ignore') as file:
        first_line = file.readline()

    for col_name in ['x', 'y', 'B', 'height', 'time']:
        if col_name not in first_line:
            return False

    return True


def gps_decoding(gps):
    """
    Decode the gps values to the desired value format.
    From the given latitude format [ddmm.mmmmmmm] (not relative to hour),
    to degrees.minutes [dd.mmmmmmm..] (relative to hour).
    d=degrees.
    m=minutes.

    We take the first 2 digits by removing %100 of the number,
    then we convert decimal to minutes by dividing by 60,
    and finally, the number is in degrees.minutes where minutes is relative to hour now.

    Args:
        gps: The gps values.

    Returns:
        Decoded gps values.
    """
    gps_degrees_100 = gps - gps % 100
    gps_minutes = (gps - gps_degrees_100) / 60

    return gps_degrees_100 / 100 + gps_minutes


def translate_gps(lon, lat):
    """
    Function that translate the given GPS format to the requested format.

    Args:
        lon: Longitude.
        lat: Latitude.

    Returns:
        The gps in the requested format.
    """
    result_utm = utm.from_latlon(gps_decoding(lat), gps_decoding(lon))

    return result_utm[0], result_utm[1]


def gpgga_loader(path: str) -> HorizontalScan:
    """
    Load blackwidow scan txt file of format gpgga into HorizontalScan object.

    Supported formats - GPGGA

    Args:
        path: full path of blackwidow scan txt file of format gpgga.

    Returns:
        HorizontalScan object with the data of the file.
    """
    scan_df = pd.read_csv(path, names=GPGGA_COLUMNS, dtype={'time': 'str'})
    easting, northing = translate_gps(scan_df['longitude'].to_numpy(),
                                      scan_df['latitude'].to_numpy())
    time = scan_df['time'].astype(str).apply(format_time).to_numpy()

    scan = HorizontalScan(file_name=path,
                          x=easting,
                          y=northing,
                          a=scan_df['altitude'],
                          b=scan_df['magnetic_field'],
                          time=time,
                          is_base_removed=False)

    return scan


def h5_loader(path: str) -> HorizontalScan:
    """
    Load blackwidow scan h5 file into HorizontalScan object.

    Args:
        path: full path of mag scan h5 file.

    Returns:
        HorizontalScan object with the data of the file.
    """
    with h5py.File(path, 'r') as h5f:
        # Get the interpolated data group from the h5 and create InterpolatedScan if the group isn't empty.
        interpolated_group = h5f.get('interpolated_data')

        interpolated_data = None if len(interpolated_group.items()) == 0 \
            else InterpolatedScan(x=interpolated_group.get('x'),
                                  y=interpolated_group.get('y'),
                                  b=interpolated_group.get('b'),
                                  mask=interpolated_group.get('mask'),
                                  interpolation_method=interpolated_group.attrs.get('interpolation_method'))

        # Get and handle date attribute and time dataset.
        date = h5f.attrs.get('date')
        if date == 'None':
            date = None
        else:
            date = datetime.datetime.strptime(date, '%Y-%m-%d').date()

        time = pd.Series(h5f.get('time')).apply(base_format_time)

        # Create HorizontalScan object with the data from the h5 file.
        scan = HorizontalScan(file_name=h5f.attrs.get('file_name'),
                              x=h5f.get('x'),
                              y=h5f.get('y'),
                              a=h5f.get('a'),
                              b=h5f.get('b'),
                              time=time,
                              date=date,
                              is_base_removed=bool(h5f.attrs.get('is_base_removed')),
                              interpolated_data=interpolated_data)

    return scan


def lab_txt_format_loader(path: str) -> HorizontalScan:
    """
    Load blackwidow scan txt file of lab's format into HorizontalScan object.

    Supported formats - lab's format.

    Args:
        path: full path of blackwidow scan txt file of lab's format.

    Returns:
        HorizontalScan object with the data of the file.
    """
    scan_df = pd.read_csv(path, dtype={'time': 'str'}, sep='\t')

    time = scan_df['time'].astype(str).apply(format_time)

    scan = HorizontalScan(file_name=path,
                          x=scan_df['x'],
                          y=scan_df['y'],
                          a=scan_df['height'],
                          b=scan_df['B'],
                          time=time,
                          is_base_removed=True)

    return scan


def check_if_base_removed(b) -> bool:
    """Simple check for b array"""
    return bool(np.median(b) < 40e3)


def load_gz_scan(scan_path: str) -> HorizontalScan:
    """
    Load gz real scan into mag scan object.

    Args:
        scan_path: The path to the gz's txt file scan.

    Returns:
        HorizontalScan.
    """
    scan_df = pd.read_csv(scan_path, dtype={'time': 'str'}, index_col=False, sep="\t")
    time = scan_df['time'].astype(str).apply(format_time)

    is_base_removed = check_if_base_removed(scan_df['B'])

    scan = HorizontalScan(file_name=scan_path,
                          x=scan_df['x'],
                          y=scan_df['y'],
                          a=scan_df['height'],
                          b=scan_df['B'],
                          time=time,
                          is_base_removed=is_base_removed)

    return scan


def csv_loader(scan_path: str) -> HorizontalScan:
    """
    Load HorizontalScan from csv/txt file that created from our(mag_utils) saver.

    Look for structure and further information in save_as_csv.py.

    Args:
        scan_path: The path to the csv/txt.

    Returns:
        HorizontalScan object.
    """

    scan_df = pd.read_csv(scan_path, dtype={'time': str, 'date': str, 'file_name': str}, header=1, )
    if isinstance(scan_df.get('date')[0], str):
        date = datetime.datetime.strptime(scan_df.get('date')[0], '%Y-%m-%d').date()
    else:
        date = None

    scan = HorizontalScan(file_name=scan_df["file_name"][0],
                          x=scan_df["x"].to_numpy(),
                          y=scan_df["y"].to_numpy(),
                          a=scan_df["a"].to_numpy(),
                          b=scan_df["b"].to_numpy(),
                          time=scan_df["time"].dropna().apply(base_format_time).to_numpy(),
                          is_base_removed=scan_df["is_base_removed"][0],
                          date=date)
    return scan


def txt_loader(path: str) -> HorizontalScan:
    """
    Load blackwidow scan txt file into HorizontalScan object.

    Supported formats - '$GPGGA', lab's format.

    Args:
        path: full path of mag scan txt file.

    Returns:
        HorizontalScan object with the data of the file.
    """
    with open(path, 'r') as file:
        first_line = file.readline()

    if first_line.startswith('$GPGGA'):
        scan = gpgga_loader(path)
    elif first_line.startswith('HorizontalScan'):
        scan = csv_loader(path)
    elif is_gz_file(path):
        scan = load_gz_scan(path)
    elif 'original B' in first_line:
        scan = lab_txt_format_loader(path)
    else:
        raise ValueError(f'Unexpected txt file content.'
                         f"the only files we support is gpgga,gz,lab."
                         f'first line was:\n{first_line}')

    return scan


def dat_loader(path: str) -> HorizontalScan:
    """
    Load basic scan dat file into HorizontalScan object.

    Supported formats - dat format.

    Args:
        path: full path of mag scan dat file.

    Returns:
        HorizontalScan object with the data of the file.
    """
    scan_df = pd.read_table(path, sep=" ", skiprows=None, encoding='utf-8',
                            error_bad_lines=False)

    if scan_df.empty:
        raise ValueError('Empty dat file')

    scan_df['Time'] = scan_df['Time'].astype(str).apply(lambda time: time.replace(':', '')).apply(
        format_time).to_numpy()

    scan = HorizontalScan(file_name=path,
                          x=scan_df['Longitude'],
                          y=scan_df['Latitude'],
                          a=scan_df['Altitude(m)'].to_numpy(),
                          b=scan_df['G823_F'].to_numpy(),
                          time=scan_df['Time'],
                          is_base_removed=False)

    return scan


def load(path: str) -> HorizontalScan:
    """
    Load blackwidow scan file into HorizontalScan object.

    Supported file types - txt, h5, dat, csv

    Args:
        path: full path of mag scan file.

    Returns:
        HorizontalScan object with the data of the file.
    """
    if path.lower().endswith('.txt') or path.lower().endswith('.csv'):
        scan = txt_loader(path)
    elif path.lower().endswith('.dat'):
        scan = dat_loader(path)
    elif path.lower().endswith('.h5') or path.lower().endswith('.hdf5'):
        scan = h5_loader(path)
    else:
        raise ValueError(f'Unsupported file type.'
                         f' Only supporting txt/dat/h5/csv files.'
                         f' file: {path}')

    return scan
