import argparse
import os
import time
from Data.calculate_statistics import calculate_statistics, calculate_MAD
import pandas as pd

from Data.fft_on_magnetic_field import fourier_transform

current_time = time.strftime("%m.%d.%y %H:%M", time.localtime())
current_time = current_time.replace(".", "_").replace(" ", "_").replace(":", "_")
def load_txt(path):
    # load data from txt file into a pandas data frame
    data_frame = pd.read_csv(path, delim_whitespace=True, header=None)

    # extract x coordinate, y coordinate, B magnetic field and height
    x = pd.to_numeric(data_frame.iloc[1:, 0], errors='coerce').values
    y = pd.to_numeric(data_frame.iloc[1:, 1], errors='coerce').values
    B = pd.to_numeric(data_frame.iloc[1:, 2], errors='coerce').values
    height = pd.to_numeric(data_frame.iloc[1:, 3], errors='coerce').values

    return x, y, B, height

parser = argparse.ArgumentParser()
parser.add_argument("-statistics", "--statistics", help="calculate median", default=True)
#parser.add_argument("-path", "--path", help="path to txt file", default="C:/Users/97252/Documents/experiments/Matan_experiments/clear_data_original_values_GZ_20F1000_MESURE_20220804070630_polygon_A_9m.txt")
#parser.add_argument("-path", "--path", help="path to txt file", default="C:/Users/97252/Documents/experiments/Matan_experiments/clear_data_original_values_GZ_20F1000_MESURE_20220804064138_polygon_A_6.5_meter.txt")
parser.add_argument("-path", "--path", help="path to txt file", default="C:/Users/97252/Documents/experiments/Matan_experiments/clear_data_GZ_20F1000_MESURE_20220804064138_polygon_A_6.5_meter.txt")
#parser.add_argument("-out", "--out", help="output path", default=current_time)
parser.add_argument("-out", "--out", help="output path", default="C:/Users/97252/PycharmProjects/GO/src/Data/02_27_23_12_02_clear_data_6.5")
#parser.add_argument("-out", "--out", help="output path", default="C:/Users/97252/PycharmProjects/GO/src/Data/02_27_23_12_02_clear_data_original_values_6.5")

parser.add_argument("-fft", "--fft", help="manipulation with fourier transform", default=True)




if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.exists(args.out):
        os.makedirs(args.out)
    x, y, B, height = load_txt(args.path)
    if args.statistics:
        calculate_statistics(B, args.out)
        calculate_MAD(x, y, B, args.out)
    if args.fft:
        fourier_transform(x, y, B, args.out)



