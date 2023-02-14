"""The mag Utils package. Constains all the code needed to handle the mag data."""
import sys
from ._consts import mag_UTILS_VERSION
from .loader.main_load import load  # noqa: F401
from .scans import base_scan, horizontal_scan, mag_scan, labeled_scan, interpolated_scan

sys.modules['mag_utils.base_scan'] = base_scan
sys.modules['mag_utils.horizontal_scan'] = horizontal_scan
sys.modules['mag_utils.scans.mag_scan'] = mag_scan
sys.modules['mag_utils.labeled_scan'] = labeled_scan
sys.modules['mag_utils.scans.interpolated_scan'] = interpolated_scan

__version__ = mag_UTILS_VERSION
