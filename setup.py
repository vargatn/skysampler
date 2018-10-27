import sys
from setuptools import setup, find_packages

if sys.version_info[0] != 3:
    sys.exit("Only python 3 is supported at the moment")

setup(name="skysampler",
      packages=find_packages(),
      description="Generate random realizations of line-of-sights in optical sky surveys ",
      install_requires=['numpy', 'scipy', 'pandas', 'astropy', 'fitsio', "healpy"],
      author="Tamas Norbert Varga",
      author_email="T.Varga@physik.lmu.de",
      version="0.0")
