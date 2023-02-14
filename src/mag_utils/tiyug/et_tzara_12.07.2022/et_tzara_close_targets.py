from mag_utils.labeled_scan import Target
from tiyug.utils import save_and_show


def et_tzara():
    data_et_tzara_dir_input = r"../../Data/labeled_data/et_tzara/12.07.2022/original_files/close_targets/"
    data_et_tzara_dir_out = r"../../Data/labeled_data/et_tzara/12.07.2022/labeled_files/close_targets"
    labels = [Target("reshet 1", [677282.282, 3565064.266, 40.060], [0, 0, 0]),
              Target("reshet 2", [677282.255, 3565065.495, 40.111], [0, 0, 0])
              ]

    save_and_show(data_et_tzara_dir_input, labels, 'ish', data_et_tzara_dir_out)


if __name__ == '__main__':
    et_tzara()
