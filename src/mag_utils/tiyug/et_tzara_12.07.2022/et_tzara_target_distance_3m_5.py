from mag_utils.labeled_scan import Target
from tiyug.utils import save_and_show


def et_tzara():
    data_et_tzara_dir_input = r"../../Data/labeled_data/et_tzara/12.07.2022/original_files/seperate_targets_3m/"
    data_et_tzara_dir_out = r"../../Data/labeled_data/et_tzara/12.07.2022/labeled_files/seperate_targets_3m"
    labels = [Target("reshet 1", [677282.141, 3565065.539, 40.026], [0, 0, 0]),
              Target("reshet 2", [677282.468, 3565061.493, 39.903], [0, 0, 0])
              ]

    save_and_show(data_et_tzara_dir_input, labels, "ish", data_et_tzara_dir_out)


if __name__ == '__main__':
    et_tzara()
