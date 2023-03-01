import warnings
from scipy.optimize import minimize
from magrnd.ground_one.data_processing.consts import IA_CONSTANT, IA_CALIB_INITIAL_VALUES, CALIB_METHODS, WINDOW_TITLE
from magrnd.ground_one.graphics.CalibrationWindow import CalibrationWindow
import numpy as np

warnings.filterwarnings("ignore")


def find_optimized_parameters_for_calibration(GZ_cropped_calib_scan):
    bx, by, bz = GZ_cropped_calib_scan['bx'] * IA_CONSTANT, \
                 GZ_cropped_calib_scan['by'] * IA_CONSTANT, \
                 GZ_cropped_calib_scan['bz'] * IA_CONSTANT
    parameters_history = []

    def save_parameters_history(x):
        parameters_history.append(x)

    initial_parameters = IA_CALIB_INITIAL_VALUES
    for index, method in enumerate(CALIB_METHODS):
        print("\n" + method, index + 1, "/", len(CALIB_METHODS))

        optimization_obj = minimize(std_of_vec_fun, initial_parameters, args=(bx, by, bz), method=method,
                                    callback=save_parameters_history, options={'disp': False})

        optimized_parameters = optimization_obj['x']
        initial_parameters = optimized_parameters
        bx_cal, by_cal, bz_cal = calib_ish_raw_with_matrices(np.array([bx, by, bz]).T, *optimized_parameters).T

        std_val = np.std(np.sqrt(bx_cal ** 2 + by_cal ** 2 + bz_cal ** 2))
        print('std = ', std_val)
        print('calibration parameters: \n', initial_parameters)

    return initial_parameters


def std_of_vec_fun(vals, bx, by, bz):
    bx_cal, by_cal, bz_cal = calib_ish_raw_with_matrices(np.array([bx, by, bz]).T, *vals).T
    b_tot = np.sqrt(bx_cal ** 2 + by_cal ** 2 + bz_cal ** 2)
    return np.std(b_tot / np.average(b_tot))


def calib_ish_raw_with_matrices(matrix, b1, b2, b3, s1, s2, s3, u1, u2, u3):
    # create matrices
    P = np.array([[1, 0, 0],
                  [-np.sin(u1), np.cos(u1), 0],
                  [np.sin(u2), np.sin(u3), np.sqrt(1 - np.sin(u2) ** 2 - np.sin(u3) ** 2)]])

    S = np.array([[s1, 0, 0],
                  [0, s2, 0],
                  [0, 0, s3]])
    # return calibrated matrix
    return np.matmul(
        np.matmul(np.linalg.inv(P), np.linalg.inv(S)),
        (matrix - np.array([b1, b2, b3])).T
    ).T


def apply_ia_calib(matrix, calib_file_path):
    # check if a calibration path has been passed
    if calib_file_path is None:
        return matrix[["bx", "by", "bz"]].to_numpy() * IA_CONSTANT

    # check if the calibration is pre-made or it needs to be made
    if calib_file_path.name.endswith('dat'):
        with open(calib_file_path, "r") as calib_file_stream:
            calib_file_content = [val for val in calib_file_stream.read().split(" ") if len(val)][1:-1]
            b1, b2, b3, s1, s2, s3, u1, u2, u3 = tuple(map(float, calib_file_content))

    else:  # create new calibration from scratch
        calibration_window = CalibrationWindow(calib_file_path)
        cropped_data = calibration_window.get_cropped_data()

        print("Performing Calibration...")
        b1, b2, b3, s1, s2, s3, u1, u2, u3 = find_optimized_parameters_for_calibration(cropped_data)

        gz_calib_path = calib_file_path.with_name(calib_file_path.stem + '_GZ_calibration.dat')
        with open(gz_calib_path, 'w') as f:
            f.write('1\n')
            f.write('  ' + '   '.join(map(str, [b1, b2, b3, s1, s2, s3, u1, u2, u3, '\n'])))
            print(f'Calibration file saved to:\n {gz_calib_path}')
    return calib_ish_raw_with_matrices(matrix[["bx", "by", "bz"]].to_numpy() * IA_CONSTANT, b1, b2, b3, s1, s2, s3, u1,
                                       u2,
                                       u3)
