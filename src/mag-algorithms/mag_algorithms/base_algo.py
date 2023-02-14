"""Algorithms class that all class is inherit."""
from abc import ABC

from mag_utils.scans.horizontal_scan import HorizontalScan


class Algorithm(ABC):
    """The class that all algorithms inherit from him."""

    def run(self, scan: HorizontalScan, threshold=None):
        """
        Run the algorithm.

        Args:
            scan: HorizontalScan object to give as input to the algorithm.
            threshold: Threshold to make the prediction count as true.

        Returns:
            Result dict:
            {"x": numeric, "y": numeric, "d2s": numeric,
             "score": numeric, "name": string}

            Or None if there is not result.
        """
        raise NotImplementedError("y u no do tis?!!?!?")
