def identify_scan(file_path):
    with open(file_path, mode="r", encoding="ISO-8859-1") as data_file:
        lines = data_file.readlines()[:2]

    if lines[0][0] == "$" or lines[1][0] == "$":
        scan_type = "widow"
    elif lines[0][0] == "#" or lines[1][0] == "#":
        scan_type = "base"
    elif lines[0][0] == "x" or lines[1][0] == "x":
        scan_type = "gz"
    elif lines[0][0] == "V" or lines[1][0] == "V":
        scan_type = "ia_raw"
    elif lines[0][0] == "P" or lines[1][0] == "P":
        scan_type = "ia_calibrated"
    else:
        raise TypeError("Couldn't recognize file's type")

    return scan_type


if __name__ == "__main__":
    from pathlib import Path
    from tkinter.filedialog import askopenfile
    file_path = Path(
        askopenfile(title="Select file", filetypes=[(".txt", ".txt"), (".dat", ".dat")]).name)
    scan_type = identify_scan(file_path)
    print(scan_type)
