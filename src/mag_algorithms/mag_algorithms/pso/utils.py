"""Utils in the future insert to mag utils."""
import numpy as np
from mag_utils.mag_utils.scans.horizontal_scan import HorizontalScan


def normalize_mag_scan(mag_scan: HorizontalScan):
    """
    Normalize to tesla units.

    Args:
        mag_scan: mag scan.

    Returns:
        The field normalize and the route.
    """
    b_scan = mag_scan.b * 1e-9  # convert to Tesla units
    avg_scan_b = b_scan - b_scan.mean()
    scan_route = np.array([mag_scan.x, mag_scan.y, mag_scan.a]).T

    return avg_scan_b, scan_route
