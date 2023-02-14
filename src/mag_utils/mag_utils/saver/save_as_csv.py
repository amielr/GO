"""Function that save object data in csv"""
from time import gmtime, strftime
import numpy as np
import pandas as pd




def save(scan, output_path=None):
    """
    Save the data in csv file.

    The first cell in the first row of the CSV file is the type of scan it is.
    next we have the rest of the csv where each element is a column.

    Args:
        scan: Object of magScanBase, InterpolatedScan, or BaseScan (Can work with any other object).
        output_path: The output file path.
    """

    data_dict = {}
    if scan is None:
        raise ValueError("Scan can't be None.")

    for attr_name, attr_value in scan.__dict__.items():
        if isinstance(attr_value, np.ndarray):
            data_dict[attr_name] = pd.Series(attr_value.astype(
                str).flatten().tolist() if attr_name == "time" else attr_value.flatten().tolist())
        elif attr_name == "interpolated_data":
            continue
        # Add int, bool, str or float as columns
        elif isinstance(attr_value, (int, bool, str, float)):
            data_dict[attr_name] = pd.Series(attr_value)
        # Add date and lists as string attribute.
        elif attr_name == "date" or isinstance(attr_value, list):
            data_dict[attr_name] = None if attr_value is None else str(attr_value)
        else:
            raise TypeError(f"Don't have option of saving {attr_name} of type {type(attr_value)} in the csv.")

    dataframe = pd.DataFrame.from_dict(data_dict)

    if output_path is None:
        curr_time = strftime("%Y-%m-%d_%H.%M.%S", gmtime())
        output_path = f"{scan.__class__.__name__}_{str(curr_time)}.csv"

    with open(output_path, 'w') as csv_file:
        csv_file.write(scan.__class__.__name__ + "\n")
        dataframe.to_csv(csv_file, index=False, line_terminator="\n")
