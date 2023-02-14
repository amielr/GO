"""Package build script."""

import setuptools
import mag_utils

with open("README.md", 'r') as fh:
    long_description = fh.read()

# Initialize install_requires packages' list
with open('requirements.txt', 'r') as req_file:
    install_requires = [line.strip() for line in req_file.readlines()]

VERSION = mag_utils.__version__

setuptools.setup(
    name="mag_utils",
    version=VERSION,
    author="SamurAI",
    description="All of mag utils in one package.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    test_suite='pytest',
    install_requires=install_requires,
    python_requires=">=3.6"
)
