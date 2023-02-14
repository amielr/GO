import pytest
from unittest import mock
from mag_utils.functional.validator import regex_head_base, regex_data_location_base, regex_data_base, \
    regex_data_widow, validator_line_base, validator_line_widow, validate


@pytest.fixture
def text_widow_with_wrong_line_mock():
    return "$GasdaPGGA,090109.90,3120.1056938,N,03422.5723730,E,2,1,M,17.80,M,1 sdfs8,TSTR*59, 44747.711,1470\n" \
           "$GaPsadGGA,090110.00,3120.1056922,N,03422.5723749,sdfsfE,2,17fgdg.80,M,18,TSTR*5D, 44747.488,1470\n" \
           "$GasdPGGA,09sad0110.10,sad.10sad56942,asdN,03422.5723743sdfsdE,,17.80,M,18,dsf*5F, 44747.655,1467\n" \
           "$GasdPGGA,090110.20asd,3120.1056930,N,03422.5723738,E,2,10,0.8,104.478,M,17.80,M,18,TSTR*55, 44747.678,1467\n" \
           "$GPGGasdA,090110.30,3120.1056935,N,03422.5723736,E,2,10,0.8,104.40,0.8,104.480,M,17.80,M,18,TSTR*53, " \
           "44747.605,1467\n" \
           "asdasd,090110.50,3120.1056924,N,03422.5723745,E,2,10,0.8,dsfs 1080,M,19,TSTR*59, 44747.676,1467\n"


@pytest.fixture
def text_widow_with_correct_line_mock():
    return "$GPGGA,090109.90,3120.1056938,N,03422.5723730,E,2,10,0.8,104.477,M,17.80,M,18,TSTR*59, 44747.711,1470\n" \
           "$GPGGA,090110.00,3120.1056922,N,03422.5723749,E,2,10,0.8,104.477,M,17.80,M,18,TSTR*5D, 44747.488,1470\n" \
           "$GPGGA,090110.10,3120.1056942,N,03422.5723743,E,2,10,0.8,104.478,M,17.80,M,18,TSTR*5F, 44747.655,1467\n" \
           "$GPGGA,090110.20,3120.1056930,N,03422.5723738,E,2,10,0.8,104.478,M,17.80,M,18,TSTR*55, 44747.678,1467\n" \
           "$GPGGA,090110.30,3120.1056935,N,03422.5723736,E,2,10,0.8,104.479,M,17.80,M,18,TSTR*5E, 44747.638,1470\n" \
           "$GPGGA,090110.40,3120.1056940,N,03422.5723738,E,2,10,0.8,104.480,M,17.80,M,18,TSTR*53, 44747.605,1467\n" \
           "$GPGGA,090110.50,3120.1056924,N,03422.5723745,E,2,10,0.8,104.482,M,17.80,M,19,TSTR*59, 44747.676,1467\n"


@pytest.fixture
def text_base_with_correct_line_mock():
    return """# S/N: G862DC066      
# Software Version: 01.02.03
# Build Date:       Jun 16 2015 19:29:21
# Output Baud Rate = 19200
# Mag Baud Rate = 19200
# GPS Baud Rate = 19200
# Debug Port Baud Rate = 19200
# 1 PPS Edge = negative edge
# Auto 1 PPS Edge Detect = true
# Log GPS Data with Mag Data = true
$GPRMC,170217.000,V,3305.50679,N,03508.49360,E,0.0,0.0,080121,,,N*70
$ 32721.983,0093,00:00:08.055,00/00/00,50
$ 32845.767,0095,00:00:08.155,00/00/00,50
$ 32665.659,0093,00:00:08.255,00/00/00,50
$ 32552.240,0093,00:00:08.355,00/00/00,50
$ 32771.045,0093,00:00:08.455,00/00/00,50
$ 32774.909,0093,00:00:08.555,00/00/00,50
$ 32597.465,0093,00:00:08.655,00/00/00,50"""


def test_regex_head_base():
    head_line_valid = "# Build Date:       Jun 16 2015 19:29:21"
    assert regex_head_base(head_line_valid)
    head_line_invalid = "$Build Date:       Jun 16 2015 19:29:21"
    assert not regex_head_base(head_line_invalid)


def test_regex_data_location_base():
    data_location_valid = "$GPRMC,170217.000,V,3305.50679,N,03508.49360,E,0.0,0.0,080121,,,N*70"
    assert regex_data_location_base(data_location_valid)
    data_location_invalid = ")1!19a„„ÅÅ¥¥	 J89¦…B9k	ê(ya!!1q)iH€"
    assert not regex_data_location_base(data_location_invalid)


def test_regex_data_base():
    data_line_valid = "$ 32790.053,0093,18:30:18.355,01/08/21,10"
    assert regex_data_base(data_line_valid)
    data_line_invalid = "iiòà  ¬b+!q%!á"
    assert not regex_data_base(data_line_invalid)


def test_regex_data_widow():
    data_line_widow_valid = \
        "$GPGGA,090110.80,3120.1056942,N,03422.5723750,E,2,10,0.8,104.473,M,17.80,M,09,TSTR*5F, 44747.766,1465"
    assert regex_data_widow(data_line_widow_valid)
    data_line_widow_invalid = "iiòà  ¬b+!q%!á"
    assert not regex_data_widow(data_line_widow_invalid)


def test_validator_line_base():
    head_line_valid = "# Build Date:       Jun 16 2015 19:29:21"
    data_location_valid = "$GPRMC,170217.000,V,3305.50679,N,03508.49360,E,0.0,0.0,080121,,,N*70"
    data_line_valid = "$ 32790.053,0093,18:30:18.355,01/08/21,10"
    invalid_line = "iiòà  ¬b+!q%!á"
    assert validator_line_base(head_line_valid)
    assert validator_line_base(data_location_valid)
    assert validator_line_base(data_line_valid)
    assert not validator_line_base(invalid_line)


def test_validator_line_widow():
    data_line_widow_valid = \
        "$GPGGA,090110.80,3120.1056942,N,03422.5723750,E,2,10,0.8,104.473,M,17.80,M,09,TSTR*5F, 44747.766,1465"
    invalid_line = "iiòà  ¬b+!q%!á"
    assert validator_line_widow(data_line_widow_valid)
    assert not validator_line_widow(invalid_line)


def test_no_txt_error():
    with pytest.raises(ValueError, match="The format file isn't txt"):
        validate(file_to_validate_path="file.not_txt",
                 validated_file_path="output.txt",
                 log_file_path="log.log",
                 file_type="widow")


def test_validate_invalid_widow(text_widow_with_wrong_line_mock):
    # check if when widow file is invalid the log file is full

    method_path = "mag_utils.functional.validator"

    open_mock = mock.mock_open(read_data=text_widow_with_wrong_line_mock)
    expected_input_writelines = text_widow_with_wrong_line_mock.splitlines(keepends=True)

    with pytest.raises(ValueError, match="No valid rows found"):
        with mock.patch(f"{method_path}.open", open_mock) as m:
            validate(file_to_validate_path="file.txt",
                     validated_file_path="out.txt",
                     log_file_path="bad_lines_widow.log",
                     file_type="widow")
            m().writelines.assert_any_call(expected_input_writelines)

    # check if when widow file is invalid and to_log is false nothing is written
    with pytest.raises(ValueError, match="No valid rows found"):
        with mock.patch(f"{method_path}.open", open_mock) as m:
            validate(file_to_validate_path="file.txt",
                     validated_file_path="out.txt",
                     log_file_path="bad_lines_widow.log",
                     file_type="widow", to_log=False)
            m().writelines.assert_not_called()


def test_validate_valid_widow(text_widow_with_correct_line_mock):
    method_path = "mag_utils.functional.validator"

    # check if when widow file is valid
    open_mock = mock.mock_open(read_data=text_widow_with_correct_line_mock)
    expected_input_writelines = text_widow_with_correct_line_mock.splitlines(keepends=True)

    with mock.patch(f"{method_path}.open", open_mock) as m:
        validate(file_to_validate_path="file.txt",
                 validated_file_path="out.txt",
                 log_file_path="correct_lines_widow.log",
                 file_type="widow")
        m().writelines.assert_any_call([])
        m().writelines.assert_any_call(expected_input_writelines)


def test_validate_valid_base(text_base_with_correct_line_mock):
    method_path = "mag_utils.functional.validator"

    # check if when base file is valid
    open_mock = mock.mock_open(read_data=text_base_with_correct_line_mock)
    expected_input_writelines = text_base_with_correct_line_mock.splitlines(keepends=True)

    with mock.patch(f"{method_path}.open", open_mock) as m:
        validate(file_to_validate_path="file.txt",
                 validated_file_path="out.txt",
                 log_file_path="correct_lines_base.log",
                 file_type="base")
        m().writelines.assert_any_call([])
        m().writelines.assert_any_call(expected_input_writelines)
