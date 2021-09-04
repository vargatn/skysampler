from __future__ import print_function
import numpy as np
import skysampler.indexer as indexer
import skysampler.paths as paths
import glob
import os

import argparse

parser = argparse.ArgumentParser(description='Runs MultiIndexer in parallel')
parser.add_argument("--ibin", type=int, default=-1)
parser.add_argument('--noclust', action="store_true")
parser.add_argument('--norands', action="store_true")
parser.add_argument("--convert", action="store_true")
parser.add_argument("--collate", action="store_true")


if __name__ == "__main__":
    args = parser.parse_args()

