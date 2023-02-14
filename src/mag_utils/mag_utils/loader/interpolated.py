"""All the functions needed to load interpolated data from tiff files."""
import json
from tifffile import tifffile

from mag_utils.mag_utils.scans.interpolated_scan import InterpolatedScan


def tiff_loader(path: str) -> InterpolatedScan:
    """
    Load interpolated scan tiff file into InterpolatedScan object.

    Args:
        path: full path of interpolated scan file.

    Returns:
        InterpolatedScan object with the data of the file.
    """
    with tifffile.TiffFile(path) as tiff:
        metadata = json.loads(tiff.pages[0].tags['ImageDescription'].value)

        interp_method = metadata['interpolation_method']
        indexes = metadata['array_indexes_values']
        data = tiff.asarray()

        interpolated_scan = InterpolatedScan(x=data[indexes['x']],
                                             y=data[indexes['y']],
                                             b=data[indexes['b']],
                                             mask=data[indexes['mask']],
                                             interpolation_method=interp_method)

    return interpolated_scan


def json_loader(path: str) -> InterpolatedScan:
    """
    Load interpolated scan json file into InterpolatedScan object.

    Args:
        path: full path of interpolated scan file.

    Returns:
        InterpolatedScan object with the data of the file.
    """
    with open(path, "r") as json_file:
        json_obj = json.load(json_file)

        interpolated_scan = InterpolatedScan(x=json_obj['x'],
                                             y=json_obj['y'],
                                             b=json_obj['b'],
                                             mask=json_obj['mask'],
                                             interpolation_method=json_obj['interpolation_method'])

    return interpolated_scan


def load(path: str) -> InterpolatedScan:
    """
    Load interpolated scan file into InterpolatedScan object.

    Supported file types - tiff, json.

    Args:
        path: full path of interpolated scan file.

    Returns:
        InterpolatedScan object with the data of the file.
    """
    if path.lower().endswith('.tif'):
        scan = tiff_loader(path)
    elif path.lower().endswith('.json'):
        scan = json_loader(path)

    else:
        raise ValueError(f'Unsupported file type.'
                         f' Only supporting tiff and json files.'
                         f' file: {path}')

    return scan
