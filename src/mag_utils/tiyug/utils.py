import os

import numpy as np
from scipy.spatial import Delaunay
from mag_utils.scans.labeled_scan import LabeledScan
from mag_utils.loader import blackwidow

DEG2RAD = np.pi / 180


def save_and_show(path_dir, labels, sensor_type, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for path in os.listdir(path_dir):
        extend_path = os.path.join(path_dir, path)
        mag = blackwidow.load(extend_path)

        # only label the file with targets that appear in its convex hull
        convex_hull = Delaunay(np.array((mag.x, mag.y)).T)
        labels_in_hull = [target for target in labels if convex_hull.find_simplex(target.pos[:2]) >= 0.]
        labeled_scan = LabeledScan.label(mag, labels_in_hull, sensor_type, is_real=True)
        print("B:", labeled_scan.b[0], "is_base_removed:", labeled_scan.is_base_removed)
        labeled_scan.plot()

        labeled_scan.save(os.path.join(out_dir, f'{path[:-4]}_labeled.h5'))
