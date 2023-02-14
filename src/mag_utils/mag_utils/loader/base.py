"""All the functions needed to load base files."""
import datetime
import io
import h5py

import pandas as pd

from mag_utils.mag_utils.scans.base_scan import BaseScan
BASE_COLUMNS = ('magnetic_field',  # [nT]
                'signal_strength',  # some computation of the SNR
                'time',  # hh:mm:ss.sss
                'date',  # MM/DD/YY
                'unknown'  # 17.8.22 checked with the lab. they don't know. :(
                )


def format_time(time: str) -> datetime.time:
    """
    Converts a time-stamp string from 'hh:mm:ss.sss' datatime.time object

    Args:
        time: time in format 'hh:mm:ss.sss'

    Returns:
        datatime.time object
    """
    ms_s = time[8:]  # Get the '.sss'
    microsecond = int(float(ms_s) * 1_000_000) if ms_s and ms_s != "." else 0

    formatted_time = datetime.time(hour=int(time[:2]),
                                   minute=int(time[3:5]),
                                   second=int(time[6:8]),
                                   microsecond=microsecond)

    return formatted_time


def txt_loader(path: str) -> BaseScan:
    """
    Load base txt file into BaseScan object.

    Args:
        path: full path of base txt file.

    Returns:
        BaseScan object with the data of the file.
    """
    # Extract from the file only the lines that start with '$ '.
    relevant_lines = "".join([line for line in open(path)
                              if line.startswith('$ ')])
    base_df = pd.read_csv(io.StringIO(relevant_lines), sep=',', names=BASE_COLUMNS, dtype={'unknown2': 'str'})

    # Remove the dollar sign and format the time.
    base_df['magnetic_field'] = pd.to_numeric(base_df['magnetic_field'].str[2:])
    time = base_df['time'].astype(str).apply(format_time)

    base_scan = BaseScan(file_name=path,
                         b=base_df['magnetic_field'],
                         time=time)

    return base_scan


def h5_loader(path: str) -> BaseScan:
    """
    Load base h5 file into BaseScan object.

    Args:
        path: full path of base h5 file.

    Returns:
        BaseScan object with the data of the file.
    """
    with h5py.File(path, 'r') as h5f:
        # Get and handle date attribute and time dataset.
        date = h5f.attrs.get('date')
        date = date if date != 'None' else None
        time = pd.Series(h5f.get('time')).apply(format_time)

        base_scan = BaseScan(file_name=h5f.attrs.get('file_name'),
                             b=h5f.get('b'),
                             time=time,
                             date=date)

    return base_scan


def load(path: str) -> BaseScan:
    """
    Load base scan file into BaseScan object.

    Supported file types - .txt, .h5

    Args:
        path: full path of base scan file.

    Returns:
        BaseScan object with the data of the file.
    """
    if path.lower().endswith('.txt'):
        scan = txt_loader(path)
    elif path.lower().endswith('.h5') or path.lower().endswith('.hdf5'):
        scan = h5_loader(path)
    else:
        raise ValueError(f'Unsupported file type.'
                         f' Only supporting txt and h5 files.'
                         f' file: {path}')

    return scan
