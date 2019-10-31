import sys
import os
from setuptools import setup, find_packages

# if sys.version_info[0] != 3:
#     sys.exit("Only python 3 is supported at the moment")

setup(name="skysampler",
      packages=find_packages(),
      description="Generate random realizations of line-of-sights in optical sky surveys ",
      install_requires=['numpy', 'scipy', 'pandas', 'astropy'],
      author="Tamas Norbert Varga",
      author_email="T.Varga@physik.lmu.de",
      version="0.1")

# create user project file
home_path = os.path.expanduser("~")
project_path = os.getcwd()
with open(home_path + "/.skysampler.yaml", "w+") as file:
    message = "project_path: " + project_path + "/"
    file.write(message)
