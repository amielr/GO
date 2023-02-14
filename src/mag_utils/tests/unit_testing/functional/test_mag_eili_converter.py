from mag_utils.functional.mag_eili_converter import magmap_file_reformatter, normalize_column_name
import pytest
from pathlib import Path
from mag_utils.loader.main_load import load


@pytest.fixture
def mock_mag_eili_text():
    return """   GPS_X        GPS_Y       Left_X       Left_Y      Right_X      Right_Y     QUAL_IND          DOP       HEIGHT      G-858_1      G-858_2         LINE         MARK         TIME         DATE 
   637467.900 3472035.950   637469.889 3472036.156   637465.911 3472035.744            1         0.80       45.760 44864.2000000 44871.9320000           0           1 17:47:29.200     08/29/22
   637467.903 3472035.921   637469.893 3472036.122   637465.913 3472035.720            1         0.80       45.760 44864.3860000 44871.7580000           0           1 17:47:29.300     08/29/22
   637467.906 3472035.892   637469.896 3472036.088   637465.916 3472035.696            1         0.80       45.760 44864.2110000 44871.9440000           0           1 17:47:29.400     08/29/22
   637467.909 3472035.863   637469.900 3472036.054   637465.918 3472035.672            1         0.80       45.760 44864.3450000 44872.0120000           0           1 17:47:29.500     08/29/22
   637467.912 3472035.834   637469.903 3472036.020   637465.921 3472035.648            1         0.80       45.760 44864.3340000 44871.8460000           0           1 17:47:29.600     08/29/22
   637467.915 3472035.805   637469.907 3472035.986   637465.923 3472035.624            1         0.80       45.760 44864.3260000 44871.7870000           0           1 17:47:29.700     08/29/22
   637467.918 3472035.776   637469.910 3472035.952   637465.926 3472035.600            1         0.80       45.760 44864.4550000 44871.8770000           0           1 17:47:29.800     08/29/22
   637467.921 3472035.747   637469.914 3472035.918   637465.928 3472035.576            1         0.80       45.760 44864.5070000 44871.9550000           0           1 17:47:29.900     08/29/22
   637467.924 3472035.718   637469.917 3472035.884   637465.931 3472035.552            1         0.80       45.760 44864.5150000 44871.8910000           0           1 17:47:30.000     08/29/22
   637467.927 3472035.689   637469.920 3472035.850   637465.934 3472035.528            1         0.80       45.760 44864.4200000 44871.8310000           0           1 17:47:30.100     08/29/22
   637467.930 3472035.660   637469.924 3472035.816   637465.936 3472035.504            1         0.80       45.760 44864.2500000 44871.7940000           0           1 17:47:30.200     08/29/22
   637467.932 3472035.625   637469.926 3472035.783   637465.938 3472035.467            1         0.80       45.760 44864.4160000 44871.8870000           0           1 17:47:30.300     08/29/22"""


def test_magmap_file_reformatter(mock_mag_eili_text):
    # open file and write data
    mock_mag_eili_file_path = Path("mock_mag_eili.dat")
    mock_mag_eili_file = mock_mag_eili_file_path.open(mode="w")
    mock_mag_eili_file.write(mock_mag_eili_text)
    mock_mag_eili_file.close()

    # reformat mag eili file
    formatted_file_path = magmap_file_reformatter(mock_mag_eili_file_path)
    mock_mag_eili_file_path.unlink()

    # verify loading as widow
    try:
        load(str(formatted_file_path), validate=True)
    except Exception as e:
        raise e
    finally:
        formatted_file_path.unlink()


def test_normalize_column_name():
    EXPECTED_COL_NAMES = ['left_x', 'left_y', 'right_x', 'right_y', 'left_b_x', 'left_b_y', 'right_b_x', 'right_b_y', 'gps_x', 'gps_x']

    columns = ['Left_X', 'Left_Y', 'Right_X', 'Right_Y',
                           'left_sensor_X', 'left_sensor_Y', 'right_sensor_X', 'right_sensor_Y', 'GPS_X', 'GPS_X']

    normalized_cols = [normalize_column_name(col) for col in columns]

    assert normalized_cols == EXPECTED_COL_NAMES, "Column names are not normalized properly"


