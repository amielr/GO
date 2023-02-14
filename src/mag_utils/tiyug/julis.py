import numpy as np
from mag_utils.scans.labeled_scan import Target
from tiyug.utils import save_and_show, DEG2RAD


# aluminum man scan.

def julis_first_part():
    data_julis_first_dir = "../Data/labeled_data/julis/first_part/original_files"
    data_julis_first_dir_out = "../Data/labeled_data/julis/first_part/labeled_files"
    labels = [Target("plate", [658587.7, 3507944.0, 65.459], [np.sin(299 * DEG2RAD), np.cos(299 * DEG2RAD), 0]),
              Target("reshet 1", [658585, 3507901, 65.85903], [np.sin(293 * DEG2RAD), np.cos(293 * DEG2RAD), 0]),
              Target("reshet 2", [658557, 3507891, 65.6688], [np.sin(265 * DEG2RAD), np.cos(265 * DEG2RAD), 0])
              ]

    save_and_show(data_julis_first_dir, labels, 'ish', data_julis_first_dir_out)


def julis_second_part():
    data_julis_dir = "../Data/labeled_data/julis/second_part/original_files"
    data_julis_dir_out = "../Data/labeled_data/julis/second_part/labeled_files"

    labels = [Target("plate", [658587.7, 3507944.0, 65.459], [np.sin(299 * DEG2RAD), np.cos(299 * DEG2RAD), 0]),
              Target("baznat 1", [658571.3, 3507946, 65.34707], [1, 0, 0]),
              Target("baznat 2", [658566.2, 3507914, 66.80545], [0, 0, 1]),
              Target("baznat 3", [658567.7, 3507913, 67.19926], [0, 0, 1]),
              Target("baznat 4", [658572.8, 3507958, 66.47796], [np.sin(85 * DEG2RAD), np.cos(85 * DEG2RAD), 0]),
              Target("baznat 5", [658548.4, 3507949, 65.71113], [np.sin(40 * DEG2RAD), np.cos(40 * DEG2RAD), 0]),
              Target("baznat 6", [658541.8, 3507948, 65.03095], [1, 0, 0])]

    save_and_show(data_julis_dir, labels, 'ish', data_julis_dir_out)


if __name__ == '__main__':
    julis_first_part()
    julis_second_part()
