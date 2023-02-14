import glob
import os

import numpy as np

from mag_utils.scans.labeled_scan import Target
from mag_utils.loader import blackwidow
from tiyug.utils import save_and_show


def estimate_ground_height(dir_path):
    print('Estimating ground height...')
    ground_levels = []
    for path in glob.glob(os.path.join(dir_path, "*.txt")):
        filename = os.path.split(path)[-1]
        if filename.startswith("All"):
            continue

        sensor_height_above_ground = int(filename.split("_")[0])

        scan = blackwidow.load(path)
        mean_flight_altitude = scan.a.mean()
        curr_ground_level = mean_flight_altitude - sensor_height_above_ground
        ground_levels.append(curr_ground_level)

        # print(f"{filename} - mean_flight_altitude:{mean_flight_altitude}, curr_ground_level:{curr_ground_level}")

    ground_level = float(np.mean(ground_levels))
    print(f"ground_level {ground_level} +- {np.std(ground_levels)}")

    return ground_level


def zit_velimon():
    data_zit_velimon_dir = "../Data/labeled_data/zit_velimon/original_files"
    data_zit_velimon_dir_out = "../Data/labeled_data/zit_velimon/labeled_files"

    ground_level = estimate_ground_height(data_zit_velimon_dir)

    # height was estimated
    labels = [Target("tt107", [680214.4, 3554476.9, ground_level], [0, 0, 0]),
              Target("tt122", [680191.3, 3554501.1, ground_level], [0, 0, 0]),
              Target("p", [680168.9, 3554528.9, ground_level], [0, 0, 0]),
              Target("t107", [680206.4, 3554537.3, ground_level], [0, 0, 0])]

    save_and_show(data_zit_velimon_dir, labels, 'widow', data_zit_velimon_dir_out)


if __name__ == "__main__":
    zit_velimon()
