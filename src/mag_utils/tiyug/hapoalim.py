from mag_utils.scans.labeled_scan import Target
from tiyug.utils import save_and_show


# aluminum man scan.
# TODO: label is best guess. get better labels


def hapoalim():
    data_hapoalim_dir = r"../Data/labeled_data/bank_hapoalim/original_files"
    data_hapoalim_dir_out = r"../Data/labeled_data/bank_hapoalim/labeled_files"
    labels = [Target("", [710798.719, 3663071.85, 455], [0, 0, 0])]

    save_and_show(data_hapoalim_dir, labels, 'ish', data_hapoalim_dir_out)


if __name__ == '__main__':
    hapoalim()
