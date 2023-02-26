"""The main object of the mag sensor, deprecated, here for backwards compatibility."""
import warnings
from .horizontal_scan import HorizontalScan


# Todo: Decide rather move it to horizontal_scan module or not.
class magScan:
    """Replace construction to HorizontalScan for backwards compatibility."""  # TODO: add deprecated like in Simph(def)

    def __new__(cls, *args, **kwargs):
        warnings.warn("magScan will be deprecated in the future, please use <HorizontalScan> instead.")

        return HorizontalScan(*args, **kwargs)