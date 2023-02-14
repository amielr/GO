from mag_utils.scans.labeled_scan import Target
from tiyug.utils import save_and_show


# blackwidow scan
# TODO: label is best guess. get better labels

def palmachim():
    data_palmachim_dir = "../Data/labeled_data/Palmhim/original_files"
    data_palmachim_dir_out = "../Data/labeled_data/Palmhim/labeled_files"
    labels = [Target("room", [659197.8786, 3527584.971, 20.14], [0, 0, 0])]

    save_and_show(data_palmachim_dir, labels, 'widow', data_palmachim_dir_out)


if __name__ == "__main__":
    palmachim()
