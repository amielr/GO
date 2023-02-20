import glob
from pathlib import Path


def fix_encoding(*paths):
    for path in paths:
        # convert to pathlib.Path
        if isinstance(path,str):
            path= Path(path)

        # if path undefined, continue
        if path is None:
            continue

        # open file with regular encoding
        with path.open('r', encoding='utf-8') as file:
            try:
                file_text = file.read()
            except UnicodeDecodeError:
                print("Could not read file using utf-8 encoding, falling back to mag_utils.mag_utils.mag_utils.mag_utils reader...")
                return

        # if no problem with encoding, continue
        if file_text.isascii():
            continue

        # apply new encoding
        encoded_text = file_text.encode('ascii', 'ignore').decode()
        with path.open('w') as encoded_file:
            encoded_file.write(encoded_text)


if __name__ == '__main__':
    files = glob.glob(
        "<insert path here>")
    fix_encoding(*files)
