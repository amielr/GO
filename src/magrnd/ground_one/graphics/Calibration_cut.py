import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def start_calibration_search(contents):
    zerocrossdata, indexes = implement_callibration_cutting_useing_zerocross_method(contents)
    # plot_xyz(zerocrossdata[0], zerocrossdata[1], zerocrossdata[2])
    return zerocrossdata, indexes


def implement_callibration_cutting_useing_zerocross_method(data):
    indexes = apply_zero_crossed_method_get_indexes(data)
    cut_field = cut_field_based_on_indexes(indexes, data)
    return cut_field, indexes


def apply_zero_crossed_method_get_indexes(stripped_fields):
    zero_crossed_points = [determine_zero_crossing(field) for field in stripped_fields]

    index_list = [find_indexes_of_repetition(points) for points in zero_crossed_points if len(points) > 2]
    index_list = [item for sublist in index_list for item in sublist]
    print("indexlist ", index_list)
    minimumindex = min(index_list)
    maximumindex = max(index_list)
    return [minimumindex, maximumindex]



def determine_zero_crossing(field):
    zero_crossings = np.where(np.diff(np.sign(field)))[0].astype(np.int64)
    print("our zero crossings are: ", zero_crossings)
    return zero_crossings


def apply_moving_average_filter(field):
    holderfield = pd.Series(moving_average(pd.Series.to_numpy(field), 25))
    return holderfield

def find_indexes_of_repetition(zero_crossings):
    differences = np.diff(zero_crossings)
    print("differences:", differences)
    diff_of_diff = np.diff(differences)
    print("diff of diff:", diff_of_diff)
    difference_of_differences_abs = np.absolute(diff_of_diff)

    thresh = 11
    thresholding = [0 if a_ < thresh else 1 for a_ in difference_of_differences_abs]
    print("thresholding: ", thresholding)
    convolved = np.convolve(thresholding, (1, 1, 1), 'valid')
    convolvedend = np.convolve(thresholding, (1, 1, 1), 'full')
    # determine_if_calibration_is_1_sin_or_2_sins(convolved, convolvedend)
    resultbegin = np.where(convolved == 0)
    resultend = np.where(convolvedend == 0)
    print("convolved", convolved)
    print("indexes of 0:", resultbegin, resultbegin[0][0], resultend, resultend[0][-1])
    res = np.diff(resultbegin)
    print("res is:", res)
    res = [j for sub in res for j in sub]
    print("res is:", res)

    indexbegin = zero_crossings[resultbegin[0][0]]
    indexend = zero_crossings[resultend[0][-1]]
    return indexbegin, indexend

def cut_field_based_on_indexes(indexes, field_lists):
    indexbegin, indexend = indexes[0], indexes[1]
    print("our begin index is: ", indexbegin, indexend)
    updated_field = []
    const = 50
    for field in field_lists:
        updated_field.append(field[indexbegin-const:indexend+const])
    return updated_field

    #peaks = signal.find_peaks(xfield)

def moving_average(signal, window_size):
    return np.convolve(signal, np.ones(window_size), 'valid') / window_size

def plot_xyz(xfield, yfield, zfield):
    xfield = pd.DataFrame(xfield)
    yfield = pd.DataFrame(yfield)
    zfield = pd.DataFrame(zfield)
    xfield.plot(label="X field")
    yfield.plot(label="Y field")
    zfield.plot(label="Z field")
    plt.gca().legend(('X', 'Y', 'Z'))
    plt.show()
    return