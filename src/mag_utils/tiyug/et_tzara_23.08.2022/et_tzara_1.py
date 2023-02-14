from mag_utils.labeled_scan import Target
from tiyug.utils import save_and_show


def et_tzara_widow():
    data_et_tzara_dir_input = r"../../Data/labeled_data/et_tzara/23.08.2022/original_files/type_1/widow/"
    data_et_tzara_dir_out = r"../../Data/labeled_data/et_tzara/23.08.2022/labeled_files/type_1"
    labels = [
        Target("berlingo", [677290.42824266, 3565081.31780922, 39.3624511682891], [0., 0., 0]),
        Target("mitsubishi attrage", [677321.671085838, 3565050.06623342, 39.6424345750154], [0., 0., 0]),
    ]

    save_and_show(data_et_tzara_dir_input, labels, "widow", data_et_tzara_dir_out)


def et_tzara_ia():
    data_et_tzara_dir_input = r"../../Data/labeled_data/et_tzara/23.08.2022/original_files/type_1/ia/"
    data_et_tzara_dir_out = r"../../Data/labeled_data/et_tzara/23.08.2022/labeled_files/"
    labels = [
        Target("berlingo", [677290.42824266, 3565081.31780922, 39.3624511682891], [0., 0., 0]),
        Target("mitsubishi attrage", [677321.671085838, 3565050.06623342, 39.6424345750154], [0., 0., 0]),
    ]

    save_and_show(data_et_tzara_dir_input, labels, "ish", data_et_tzara_dir_out)


if __name__ == '__main__':
    et_tzara_widow()
    et_tzara_ia()
