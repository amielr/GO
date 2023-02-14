"""All the functions needed to save interpolated data to tiff files."""
import json
from tifffile import tifffile
import numpy as np
from mag_utils.mag_utils._consts import Sequence



def save_as_tiff(output_path: str,
                 x: Sequence,
                 y: Sequence,
                 b: Sequence,
                 mask: Sequence,
                 interpolation_method: str = None):
    """
    Save the interpolation data in tiff file.
    You can access the metadata(interpolation_method) with 'tiff_page.tags['ImageDescription'].value'.

    Args:
        output_path: The output file path.
        x: scan coordinate [utm].
        y: scan coordinate [utm].
        b: magnetic field [nT].
        mask: Binary (filled) convex hull of the points.
        interpolation_method: The used interpolation method.
    """
    tifffile.imwrite(output_path,
                     data=np.array([x, y, b, mask]),
                     metadata={"interpolation_method": interpolation_method,
                               "array_indexes_values": {'x': 0, 'y': 1, 'b': 2, 'mask': 3}})


def save_as_json(output_path: str,
                 x: Sequence,
                 y: Sequence,
                 b: Sequence,
                 mask: Sequence,
                 interpolation_method: str = None):
    """
    Save the interpolation data in json file.
    You can access the data by json_file["data"], and the metadata by json_file["metadata"].

    Args:
        output_path: The output file path.
        x: scan coordinate [utm].
        y: scan coordinate [utm].
        b: magnetic field [nT].
        mask: Binary (filled) convex hull of the points.
        interpolation_method: The used interpolation method.
    """
    with open(output_path, "w") as json_file:
        json.dump({"x": x.tolist(),
                   "y": y.tolist(),
                   "b": b.tolist(),
                   "mask": mask.tolist(),
                   "interpolation_method": interpolation_method}, json_file)
