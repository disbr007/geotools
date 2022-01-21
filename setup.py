import setuptools
import json

with open("readme.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

with open("requirements.txt", "r") as fh:
    DEPENDENCIES = fh.readlines()

setuptools.setup(
    name="geotools",
    version='0.1',
    author="Jeff Disbrow",
    author_email="disbr007@umn.edu",
    description="Commonly used geospatial utilites.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/disbr007/geotools.git",
    package_dir={"":"."},
    packages=setuptools.find_packages(),
    py_modules=['gdaltools', 'gpdtools', 'kmztools', 'shapelytools'],
    install_requires=DEPENDENCIES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)