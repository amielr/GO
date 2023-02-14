from mag_utils.labeled_scan import Target
from tiyug.utils import save_and_show


def et_tzara_widow():
    data_et_tzara_dir_input = r"../../Data/labeled_data/et_tzara/23.08.2022/original_files/type_3/widow/"
    data_et_tzara_dir_out = r"../../Data/labeled_data/et_tzara/23.08.2022/labeled_files/type_3"
    labels = [
        Target("berlingo", [677303.346866694, 3565059.11171669, 39.2950978454001], [0., 0., 0]),
        Target("mitsubishi attrage", [677320.814397474, 3565060.64947239, 39.9332358657364], [0., 0., 0]),
    ]

    save_and_show(data_et_tzara_dir_input, labels, "widow", data_et_tzara_dir_out)


def et_tzara_ia():
    data_et_tzara_dir_input = r"../../Data/labeled_data/et_tzara/23.08.2022/original_files/type_3/ia/"
    data_et_tzara_dir_out = r"../../Data/labeled_data/et_tzara/23.08.2022/labeled_files/"
    labels = [
        Target("berlingo", [677303.346866694, 3565059.11171669, 39.2950978454001], [0., 0., 0]),
        Target("mitsubishi attrage", [677320.814397474, 3565060.64947239, 39.9332358657364], [0., 0., 0]),
    ]

    save_and_show(data_et_tzara_dir_input, labels, "ish", data_et_tzara_dir_out)


if __name__ == '__main__':
    et_tzara_widow()
    et_tzara_ia()
