import numpy as np
import matplotlib.pyplot as plt

def calculate_statistics(B,out):
    # Calculate the mean, median, standard deviation, variance, minimum, and maximum of the array
    mean = np.mean(B)
    median = np.median(B)
    std = np.std(B)
    var = np.var(B)
    min_value = np.min(B)
    max_value = np.max(B)

    # Print the results
    print("Mean:", mean)
    print("Median:", median)
    print("Standard deviation:", std)
    print("Variance:", var)
    print("Minimum value:", min_value)
    print("Maximum value:", max_value)

    with open(out + "/statistics", "w") as f:
        f.write(f"Mean: {mean}\n")
        f.write(f"Median: {median}\n")
        f.write(f"Standard deviation: {std}\n")
        f.write(f"Variance: {var}\n")
        f.write(f"Minimum value: {min_value}\n")
        f.write(f"Maximum value: {max_value}\n")

    # build a histogram
    x = np.arange(len(B))
    plt.hist(B, bins=400)

    # add axis labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.savefig(f"{out}/magnetic_field_B_values_histogram.png")
    plt.plot(x, B)
    plt.ylim(min_value,max_value)
    plt.yticks(np.linspace(min_value, max_value, num=6))
    plt.xlim(0, len(B))
    plt.xticks(np.linspace(0, len(B), num=6))


    plt.xlabel('x')
    plt.ylabel('B')
    plt.title('B measurements')
    plt.savefig(f"{out}/magnetic_field_B_scan.png")


def calculate_MAD(x, y, B, out):
    import numpy as np
    import matplotlib.pyplot as plt

    B = np.median(B)-B
    pos_B = B[B > 0]
    neg_B = B[B < 0]
    median_pos_B = np.median(pos_B)
    median_neg_B = np.median(neg_B)

    posMAD = np.median(np.abs(pos_B - median_pos_B))
    negMAD = np.median(np.abs(neg_B - median_neg_B))
    N_MAD = 10  # fixed

    th_pos = N_MAD * posMAD
    th_neg = -1 * N_MAD * negMAD

    ind_dipol_pos = np.where(B > th_pos)[0]
    ind_dipol_neg = np.where(B < th_neg)[0]

    dipol_data = np.full_like(B, np.nan)
    dipol_data[ind_dipol_pos] = B[ind_dipol_pos]
    dipol_data[ind_dipol_neg] = B[ind_dipol_neg]

    fig = plt.figure()
    plt.plot(dipol_data, 'ro')
    plt.plot(B, 'b.')
    plt.savefig(f"{out}/magnetic_field_B_scan_with_anomaly_1.png")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, B, 'b.')
    ax.plot(x[ind_dipol_pos], y[ind_dipol_pos], dipol_data[ind_dipol_pos], 'ro')

    # View from the default angle
    ax.view_init(elev=30, azim=-60)
    plt.savefig(f"{out}/magnetic_with_anomaly_1.png")

    # View from a different angle
    ax.view_init(elev=15, azim=-120)
    plt.savefig(f"{out}/magnetic_with_anomaly_2.png")

    # View from another angle
    ax.view_init(elev=45, azim=0)
    plt.savefig(f"{out}/magnetic_with_anomaly_3.png")



