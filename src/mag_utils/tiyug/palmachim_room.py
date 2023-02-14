from mag_utils.scans.labeled_scan import Target
from tiyug.utils import save_and_show


# blackwidow scan

def palmachim_room():
    data_palmachim_dir = "../Data/labeled_data/palmachim_room/original_files"
    data_palmachim_dir_out = "../Data/labeled_data/palmachim_room/labeled_files"
    labels = [Target("room", [659233.00, 3527636.00, 20.50], [0, 0, 0])]

    save_and_show(data_palmachim_dir, labels, 'widow', data_palmachim_dir_out)


if __name__ == "__main__":
    palmachim_room()
