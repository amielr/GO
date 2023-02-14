from mag_utils.labeled_scan import Target
from tiyug.utils import save_and_show


def et_tzara():
    data_et_tzara_dir_input = r"../../Data/labeled_data/et_tzara/12.07.2022/original_files/seperate_targets_10m/"
    data_et_tzara_dir_out = r"../../Data/labeled_data/et_tzara/12.07.2022/labeled_files/seperate_targets_10m"
    labels = [Target("reshet 1", [677282.174, 3565065.508, 39.992], [0, 0, 0]),
              Target("reshet 2", [677282.927, 3565054.531, 39.638], [0, 0, 0])
              ]

    save_and_show(data_et_tzara_dir_input, labels, "ish", data_et_tzara_dir_out)


if __name__ == '__main__':
    et_tzara()
