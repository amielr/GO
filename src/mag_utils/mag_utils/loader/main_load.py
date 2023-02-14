import glob
import os
import sys
import importlib.util
from collections import OrderedDict

from mag_utils.mag_utils.loader._scan_type_recognizer import recognize_file_type
from mag_utils.mag_utils.functional.validator import validate as validate_txt

FORBIDDEN_MODULES = ["__init__", "main_load", "_scan_type_recognizer"]  # Without .py
MOVE_TO_END = ["base"]


def get_loads():
    """
    Get available Load functions as a dict.

    It run through 'loader' folder and add the 'load' function in every module to a dict.
    (Every module except FORBIDDEN_MODULES)

    Returns:
        Dict of {<module_name>: <pointer to load function>}.
    """
    loads_dict = OrderedDict()

    for mdl in glob.glob(os.path.join(os.path.dirname(__file__), "*.py")):
        module_name = os.path.basename(mdl)[:-3]

        if module_name not in FORBIDDEN_MODULES:
            # For each module from the folder that is not forbidden load it, import it and save it's load function.
            spec = importlib.util.spec_from_file_location(module_name, mdl)
            sys.modules[module_name] = importlib.util.module_from_spec(spec)
            spec.loader.load_module(module_name)

            loads_dict[module_name] = sys.modules[module_name].load

    # Because we want a certain order, we go through the MOVE_TO_END
    # list and move the type to the end of the dict.
    for mdl in MOVE_TO_END:
        loads_dict.move_to_end(mdl)

    return loads_dict


# Load all the loaders and assign it to 'loads'
loads = get_loads()


def validation(path: str, scan_type: str):
    """
    Validate the scan by it's type, and return the paths to the validated file and the log file.
    """
    file_path_without_ext = os.path.splitext(path)[0]
    validated_file_path = file_path_without_ext + "_validated.txt"

    validate_txt(file_to_validate_path=path,
                 validated_file_path=validated_file_path,
                 file_type="widow" if scan_type == "blackwidow" else scan_type,
                 to_log=False)

    return validated_file_path


def _is_raw_widow(path: str) -> bool:
    """checks if a txt file is a raw widow format (gpgga)"""
    if not path.lower().endswith('.txt'):
        return False

    with open(path, 'r', errors='ignore') as file:
        for _ in range(20):
            line = file.readline()
            if line.startswith('$GPGGA'):
                return True

    return False


def load(path: str, scan_type: str = None, validate: bool = False, save_validated_file: bool = False):
    """
    Main load function.

    This function loads the scan with the appropriate loader, it can rather get the name
    of this loader or recognize it by itself by run through the loaders and find the working one.

    Args:
        path: The path to the scan, accepting all the accepted types in the loaders.
        scan_type: Optional, the type of the scan, for example: blackwidow, base...
        validate: Rather to validate the file before loading or not.
        save_validated_file: Rather to save the validated file or not, will be named:
                             "<file_name_without_extension>_validated.txt",
                             and the log file: "<file_name_without_extension>_validation_logs.log"

    Returns:
        The Scan object or None if it didn't recognize the type.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("File not found, check the path please!")

    if scan_type is not None and scan_type not in loads.keys():
        raise ValueError(f"Got unknown scan type, valid types: <{', '.join(loads.keys())}>.Got {scan_type} instead.")

    if scan_type is None:
        scan_type = recognize_file_type(path)

    if scan_type == "blackwidow":
        validate = _is_raw_widow(path)

    if validate and path.lower().endswith(".txt"):
        path = validation(path, scan_type)

    scan = loads[scan_type](path)

    if validate and path.lower().endswith(".txt"):
        if not save_validated_file:
            if os.path.isfile(path):
                os.remove(path)

    return scan
