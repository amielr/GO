import matplotlib.pyplot as plt
import numpy as np
from mag_utils.algo_tools.np_wrappers import center_columns
from mag_utils.loader.main_load import load
from mag_utils.scans.aluminum_man import AluminumManScan

from ground_one.data_processing.consts import IA_CONSTANT


def calibrate(scan: AluminumManScan):
    """
    Relative to the frame of reference of the sensor, we have the
    following 3 vectors x', y', z' which are in the directions of each non-orthogonal
    1-axis sensor, and have lengths based on the relative gains of the sensor to the x-sensor, i.e., if a change of 1nT
    is measured as 1.01nT in the x-sensor but a change of 1nT is measured as 0.99nT in the y-sensor, the length of x'
    will be 1 and the length of y' will be 0.99/1.01. Each sensor t' measures dot(B_earth, t'). We are looking for
    orthonormal x^, y^, z^ which can be expressed as a known linear combination of x', y', z'. Without loss of
    generality we can set:
    x^ = x'
    y^ = ax' + by'
    z^ = cx' + dy' + ez',
    now we want to find a, b, c, d, e so that we'll be able to compute the magnetic field in
    each orthogonal axis so that we'll be able to find the amplitude of the magnetic field.

    We know that during the scan the amplitude of the Magnetic field should be approximately the same
    (around 45000) except for the minor changes created by the dipoles in the ground, those minor
    changes are tiny compared to the total strength of the magnetic field so we can assume they won't affect our
    estimations of a, b, c, d, e.

    The function is PROVEN to compute the OPTIMAL a, b, c, d, e that would minimize the following loss function:
        sum_{i=1}^{n} (||B_i||^2-sum_{j=1}^{n}||B_j||^2/n)^2,
        where ||B_i|| is the estimated amplitude of the magnetic field at the i'th sample computed by a linear
        combination of bx,by,bz which is determined by a, b, c, d, e, i.e.,
            Bx_i = scan.bx[i]
            By_i = scan.bx[i] * a + scan.by * b
            Bz_i = scan.bx[i] * c + scan.by * d + scan.bz * e

            B_i = (Bx_i, By_i, Bz_i).

    Lastly the function computes the gain of each sensor relative to the gian of the x-sensor, and multiplies b to have
    the mean gain (i.e., b *= (1 + ||y'|| + ||z'||) / 3), this is because the average gain is probably more correct than
    using only the x sensor's gain.
    Args:
        scan: mag scan containing non-orthogonal bx, by, bz sensor values.

    Returns:
        The amplitude of the magnetic field at each sample.
    """
    bx = IA_CONSTANT * scan.bx
    by = IA_CONSTANT * scan.by
    bz = IA_CONSTANT * scan.bz

    variables = np.column_stack([bx ** 2,
                                 by ** 2,
                                 bz ** 2,
                                 bx * by,
                                 bx * bz,
                                 by * bz])

    min_change_axis = np.linalg.svd(center_columns(variables))[2][-1]
    min_change_axis *= np.sign(min_change_axis[0])

    xx_tt, yy_tt, zz_tt, xy_tt, xz_tt, yz_tt = min_change_axis
    et = np.sqrt(zz_tt)
    dt = yz_tt / (2 * et)
    ct = xz_tt / (2 * et)
    bt = np.sqrt(yy_tt - dt ** 2)
    at = (xy_tt / 2 - ct * dt) / bt
    t = np.sqrt(xx_tt - at ** 2 - ct ** 2)

    x_opt = bx
    y_opt = bx * (at / t) + by * (bt / t)
    z_opt = bx * (ct / t) + by * (dt / t) + bz * (et / t)

    gain_y = np.linalg.norm([t, at]) / bt
    gain_z = np.linalg.norm([dt * at / bt - ct, dt * t / bt, t]) / et
    mean_gain = (1 + gain_y + gain_z) / 3

    b = np.sqrt(x_opt ** 2 + y_opt ** 2 + z_opt ** 2) * mean_gain

    return b


def try_calibration():
    fpath = "<insert path here>"
    scan = load(fpath)[3500:18000]
    plt.scatter(scan.x, scan.y, c=calibrate(scan))
    plt.colorbar()
    plt.axis("equal")
    plt.show()
