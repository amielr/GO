from pathlib import Path
import numpy as np
import os


VISUALIZATIONS_ON1_OFF0 = 0
WINDOW_TITLE = "Ground One"
IA_CONSTANT = 0.0175  # volt to Tesla
IA_CALIB_INITIAL_VALUES = np.array([-1, -1, -1,
                                    1, 1, 1,
                                    0, 0, 0])  # initial values for IA calibration
FG_COLOR = "#a6a6a6"
BG_COLOR = "#464646"
TB_COLOR = "#666666"
SARGEL_COLOR = "red"
GUI_THEME = "equilux"
SELECT_FILE_WINDOW_FONT = ("Courier", 14)
AXES_ASPECT = "equal"
HEIGHT_GRAPH_OFFSET = 3
RANGE_SELECTOR_ALPHA = 0.2
FILE_NAME_PREFIX = "GZ_"
FILE_NAME_SUFFIX = ".txt"
LINES_NAME_PREFIX = "VOLVO_LINES_"
GZ_DIR = Path.home() / ("." + WINDOW_TITLE.lower().replace(" ", "-"))
CACHE_DIR = GZ_DIR / "cache"
CACHE_LIMIT = 100
FILE_NAME_FONT = ('Gisha', 18, 'bold')
TITLE_FONT = ('Stencil', 36)
GEOTIFF_TITLE = "GEOTIFF WINDOW"
FILTER_TITLE = "FILTER WINDOW"
VOLVO_TITLE = "VOLVO\xa9 WINDOW"
INITIAL_LEVELS_NUM = 50
SAMPLE_RATE_HZ = 10
DEFAULT_LOW_PASS_FILTER_FREQS = (0.5, 1)
DEFAULT_HIGH_PASS_FILTER_FREQS = (2, 1)
APPLY_BUTTON_POS = [0.925, 0.0, 0.075, 0.05]
EXPECTED_TIMESTAMP_LEN = 6
IA_SAMPLE_RATE_HZ = 25
CALIB_METHODS = ["Powell", "Nelder-Mead", "CG", "BFGS"]  # for IA calibration
MIN_SELECTION_LENGTH = 5
HELP_TEXT = [f"Thanks for using the {WINDOW_TITLE}",
             "Please contact Ohad Kuta if you encounter any problems"
             ]
FLIP_SELECTION_SHORTCUT = "i"
TIME_RESOLUTION = 2
ANGLE_TOLERANCE = 65
CLUSTER_DISTANCE = 2
MIN_CLUSTER_SIZE = 5
VALIDATED_STR = '_validated'

VOLVO_EXPLANATION = """The VOLVO (all rights reserved \xa9) automates the cutting of scan lines, 
and averages the magnetic field values for each line.
The time resolution is the window to look at when calculating the direction of movement (in seconds).
The angle tolerance is the maximum angle at which we include the B values in the mean (in degrees).
The cluster distance is the minimum distance needed for samples to be considered in the same line.
The min cluster size is the minimum number of samples in a cluster.
"""
SAVE_AS_FILETYPES = [("Text File", "*.txt")]

SCAN_ROUTE_ALPHA = 0.1
GEO_PROJECTION = 32636
VOLVO_ENTRY_LABELS = ("Time Resolution For Computing Direction:", "Angle Tolerance Of Lines:",
                      "Distance Between Clusters:", "Minimum Number Of Samples In Cluster:")
ERROR_MSG_DOWNUP_IN_AUTO = "Auto-Pianuach can only be done on single sided scans."