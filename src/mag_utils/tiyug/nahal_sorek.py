from mag_utils.scans.labeled_scan import Target
from tiyug.utils import save_and_show


# aluminum man scan.

def nahal_sorek_lying_south():
    data_nahal_sorek = "../Data/labeled_data/nahal_sorek/original_files/lying_south"
    nahal_sorek_out = "../Data/labeled_data/nahal_sorek/labeled_files/lying_south"
    labels = [Target("plate", [675861.0, 3518141.0, 99.0], [0, -1., 0])]

    save_and_show(data_nahal_sorek, labels, 'widow', nahal_sorek_out)


def nahal_sorek_stand_east():
    data_nahal_sorek = "../Data/labeled_data/nahal_sorek/original_files/stand_east"
    nahal_sorek_out = "../Data/labeled_data/nahal_sorek/labeled_files/stand_east"

    labels = [Target("plate", [675861.0, 3518141.0, 99.0], [0., -1., 1.])]

    save_and_show(data_nahal_sorek, labels, 'widow', nahal_sorek_out)


def nahal_sorek_stand_south():
    data_nahal_sorek = "../Data/labeled_data/nahal_sorek/original_files/stand_south"
    nahal_sorek_out = "../Data/labeled_data/nahal_sorek/labeled_files/stand_south"

    labels = [Target("plate", [675861.0, 3518141.0, 99.0], [0., -1., 1.])]

    save_and_show(data_nahal_sorek, labels, 'widow', nahal_sorek_out)


if __name__ == '__main__':
    nahal_sorek_lying_south()
    nahal_sorek_stand_east()
    nahal_sorek_stand_south()
