from mag_utils.scans.labeled_scan import Target
from tiyug.utils import save_and_show


def et_tzara():
    data_et_tzara_dir_input = r"../../Data/labeled_data/et_tzara/12.07.2022/original_files/scattered_targets/"
    data_et_tzara_dir_out = r"../../Data/labeled_data/et_tzara/12.07.2022/labeled_files/scattered_targets"
    labels = [
        Target("reshet 1", [677274.760, 3565044.164, 39.618], [0, 0, 0]),
        Target("reshet 2", [677301.067, 3565046.779, 39.973], [0, 0, 0]),
        Target("reshet 3", [677282.149, 3565065.307, 40.047], [0, 0, 0])
    ]

    save_and_show(data_et_tzara_dir_input, labels, "ish", data_et_tzara_dir_out)


if __name__ == '__main__':
    et_tzara()
