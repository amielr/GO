"""Scan type recognizer."""
import h5py

from mag_utils.mag_utils.loader.blackwidow import is_gz_file

BASE_LINE_SEARCH_LIMIT = 20


def h5_recognizer(path):
    """
    H5 file type recognizer.

    It recognize the type by specific signature for each type.

    Args:
        path: The path of the scan file.

    Returns:
        Scan type or None if didn't recognize.
    """
    with h5py.File(path, 'r') as h5f:
        if h5f.get('targets') is not None:
            return "label_scan"
        if 'bx' in h5f.keys():
            return "aluminum"
        if h5f.get('interpolated_data') is not None:
            return "blackwidow"
        if h5f.get('b') is not None:
            return "base"

    return None


def txt_recognizer(path):  # pylint: disable=too-many-return-statements
    """
    Txt/csv file type recognizer.

    It recognize the type by specific signature for each type.

    Args:
        path: The path of the scan file.

    Returns:
        Scan type or None if didn't recognize.
    """
    with open(path, 'r', errors='ignore') as file:
        first_line = file.readline()

        if first_line.startswith('HorizontalScan') or \
                first_line.startswith('magScan') or \
                first_line.startswith('$GPGGA') or \
                'original B' in first_line or \
                is_gz_file(path):
            return "blackwidow"

        if first_line.startswith("VERSION"):
            return "aluminum"

        if first_line.startswith("# S/N:"):
            return "base"

        # Run through BASE_LINE_SEARCH_LIMIT first lines and search for base's signature.
        for _ in range(BASE_LINE_SEARCH_LIMIT):
            line = file.readline()

            if line.startswith("$ "):
                return "base"
            if line.startswith('$GPGGA'):
                return "blackwidow"

    return None


def recognize_file_type(path: str) -> str:
    if path.lower().endswith('.h5') or path.lower().endswith('.hdf5'):
        scan_type = h5_recognizer(path)
    elif path.lower().endswith(".txt") or path.lower().endswith(".csv"):
        scan_type = txt_recognizer(path)
    elif path.lower().endswith(".json") or path.lower().endswith(".tiff") or path.lower().endswith(".tif"):
        scan_type = "interpolated"
    else:
        raise ValueError(f'Unsupported file type.'
                         f' Only supporting txt/h5/csv/json/tiff files.'
                         f' file: {path}')

    if scan_type is None:
        raise TypeError(f"could not recognize file type on file {path}")

    return scan_type
