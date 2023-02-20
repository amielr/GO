from unittest import mock

import pytest
from mag_utils.mag_utils.loader.aluminum import get_samples_from_file


@pytest.fixture
def mock_aluminum_file_text():
    return """
VERSION 20F1 : 1.2.05 Frequency : 25.00 Hz	Filter : 36
[CHANNEL] [NUMERO_TRAME] [INDICE_GPS] [FLAG] [TEMPERATURE1] [X1] [Y1] [Z1]
>G01 GNGGA,122144.00,3141.76088,N,03440.45308,E,1,08,1.35,63.3,M,16.8,M,,*7B
>M01 00000000 00000000 0 +21.7 +0211666 +0211004 +0219592
>M01 00000001 00000000 0 +21.7 +0211594 +0210920 +0219518
>M01 00000002 00000000 0 +21.7 +0210658 +0209992 +0218586
>M01 00000003 00000000 0 +21.7 +0367780 -1819270 -1809632"""


def test_get_samples_from_file(mock_aluminum_file_text):
    method_path = "mag_utils.mag_utils.mag_utils.mag_utils.loader.aluminum"

    open_mock = mock.mock_open(read_data=mock_aluminum_file_text)

    with mock.patch(f"{method_path}.open", open_mock):
        assert len(get_samples_from_file(open_mock)) == 4

