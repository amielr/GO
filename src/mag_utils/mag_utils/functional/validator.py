import re

import numpy as np

from mag_utils.mag_utils.functional._regex import REG_CHECK_NUMBER, REG_CHECK_HIGHER_LETTER, REG_CHECK_TIME, REG_CHECK_DATE, \
    REG_widow_FILE


def regex_head_base(head_line: str) -> bool:
    """
    Check if the head line (from base file) pass the regular expression.

    Args:
        head_line: the head line.

    Returns:
        True if pass False else.
    """
    return re.search("^# ", head_line) is not None


def regex_data_location_base(line: str) -> bool:
    """
    Check if the location line (from base file) pass the regular expression.

    Args:
        line: the location line.

    Returns:
        True if pass False else.

    """
    regex = f"^[$]GPRMC,{REG_CHECK_NUMBER},{REG_CHECK_HIGHER_LETTER},{REG_CHECK_NUMBER},{REG_CHECK_HIGHER_LETTER}," \
            f"{REG_CHECK_NUMBER},{REG_CHECK_HIGHER_LETTER},{REG_CHECK_NUMBER},{REG_CHECK_NUMBER},{REG_CHECK_NUMBER}" \
            f",,,{REG_CHECK_HIGHER_LETTER}[*][0-9A-Z][0-9A-Z]"

    return re.search(regex, line) is not None


def regex_data_base(data_line: str) -> bool:
    """
    Check if the data line (from base file) pass the regular expression.

    Args:
        data_line: the data line.

    Returns:
        True if pass False else.

    """
    regex = f"^[$] {REG_CHECK_NUMBER},{REG_CHECK_NUMBER},{REG_CHECK_TIME},{REG_CHECK_DATE},[0-9A-Z][0-9A-Z]"

    return re.search(regex, data_line) is not None


def regex_data_widow(line: str) -> bool:
    """
    Check if the data line (from widow file) pass the regular expression.

    Args:
        line: the data line.

    Returns:
        True if pass False else.
    """
    regex = ",".join(REG_widow_FILE.values())

    return re.search(regex, line) is not None


def validator_line_base(line: str) -> bool:
    """
    Check if the line belong and validated to base file.

    Args:
        line: data line.

    Returns:
        True if pass False else.
    """
    return regex_head_base(line) or regex_data_location_base(line) or regex_data_base(line)


def validator_line_widow(line: str) -> bool:
    """
    Check if the line belong and validated to widow file.

    Args:
        line: data line.

    Returns:
        True if pass False else.
    """
    return regex_data_widow(line)


def validate(file_to_validate_path: str, validated_file_path: str, log_file_path: str = "", file_type: str = "widow",
             to_log=True):
    """
    Get the src file that we need to validate and insert the validated file to validated_file_path.

    Supported file format: .txt.

    Args:
        file_to_validate_path: the path of file that we need to validate.
        validated_file_path: the path of the file that we want to write the validated file.
        file_type: the supported file type is base or widow or aluminum.
        log_file_path: the file we want to write the logs. Ignored if to_log if false.
        to_log: Whether or not to save a log file of the bad lines.
    """

    if not file_to_validate_path.lower().endswith('.txt'):
        raise ValueError("The format file isn't txt")

    with open(file_to_validate_path, mode='r', errors='ignore') as file:
        lines = np.asarray(file.readlines())

    # Filter the file lines - save only the valid lines.
    if file_type == "base":
        valid_lines_numbers = [index for index, line in enumerate(lines) if validator_line_base(line)]
    elif file_type == "widow":
        valid_lines_numbers = [index for index, line in enumerate(lines) if validator_line_widow(line)]
    elif file_type == "aluminum":
        valid_lines_numbers = [index for index, line in enumerate(lines)]
    else:
        raise ValueError(f"Invalid file type | {file_type} is not recognized.")

    valid_lines = lines[valid_lines_numbers]

    if to_log:
        invalid_lines_numbers = list(set(range(len(lines))) - set(valid_lines_numbers))
        invalid_lines = lines[invalid_lines_numbers]

        with open(log_file_path, 'w') as log_file:
            log_file.writelines(invalid_lines.tolist())

    if len(valid_lines) == 0:
        raise ValueError("No valid rows found in the file (valid file has't saved)")

    # Create the new validated file and write the valid lines to it.
    with open(validated_file_path, mode='w') as validated_file:
        validated_file.writelines(valid_lines.tolist())
