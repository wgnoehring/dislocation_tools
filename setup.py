import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dislocation_tools", 
    version="0.1.0",
    author="Wolfram Georg Nöhring",
    author_email="wolfram.noehring@imtek.uni-freiburg.de",
    description="Functions for inserting dislocations into crystals",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/wgnoehring/dislocation_tools",
    packages=[
        "dislocation_tools", 
        "dislocation_tools.io", 
        "dislocation_tools.backend", 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
