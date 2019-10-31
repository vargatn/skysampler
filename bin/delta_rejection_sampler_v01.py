from __future__ import print_function, division
import fitsio as fio
import numpy as np
import pandas as pd

import time
import copy
import sys
import os

import sklearn.decomposition as decomp
import skysampler.emulator as emulator
import matplotlib as mpl
import subprocess as sp
import scipy.interpolate as interpolate

try:
    import cPickle as pickle
except:
    import pickle

import multiprocessing as mp

tag = "delta_rejection_sample_v01"
NSAMPLES = 10000
NCHUNKS = 16
CATALOG_RMAX = 5.
DRAW_RMAX = 3.

root_path = "/e/eser2/vargatn/EMULATOR/DELTA/resamples/"
wide_data_path = "/e/eser2/vargatn/EMULATOR/GAMMA/multi-indexer-gamma_v001_clust__z0_l1_py2.p"
deep_data_path = "/e/eser2/vargatn/EMULATOR/DELTA/run-ugriz-mof02_naive-cleaned.fits"

master_seed=12345

deep_c_settings = {
    "columns": [
        ("COLOR_G_R", (("bdf_mag", 1), ("bdf_mag", 2), "-")),
        ("COLOR_R_I", (("bdf_mag", 2), ("bdf_mag", 3), "-")),
        ("COLOR_I_Z", (("bdf_mag", 3), ("bdf_mag", 4), "-")),
    ],
    "logs": [False, False, False],
    "limits": [(-1, 3), (-1, 3), (-1, 3)],
}
deep_smc_settings = {
    "columns": [
        ("GABS", (("bdf_g", 0), ("bdf_g", 1), "SQSUM")),
        ("SIZE", ("bdf_T", 1, "+")),
        ("FRACDEV", "bdf_fracdev"),
        ("MAG_I", ("bdf_mag", 2)),
        ("COLOR_G_R", (("bdf_mag", 1), ("bdf_mag", 2), "-")),
        ("COLOR_R_I", (("bdf_mag", 2), ("bdf_mag", 3), "-")),
        ("COLOR_I_Z", (("bdf_mag", 3), ("bdf_mag", 4), "-")),
    ],
    "logs": [False, True, False, False, False, False, False,],
    "limits": [(0., 1.), (-1, 5), (-3, 4), (17, 25.5), (-1, 3), (-1, 3), (-1, 3) ],
}

wide_cr_settings = {
    "columns": [
        ("COLOR_G_R", ("MOF_CM_MAG_CORRECTED_G", "MOF_CM_MAG_CORRECTED_R", "-")),
        ("COLOR_R_I", ("MOF_CM_MAG_CORRECTED_R", "MOF_CM_MAG_CORRECTED_I", "-")),
        ("COLOR_I_Z", ("MOF_CM_MAG_CORRECTED_I", "MOF_CM_MAG_CORRECTED_Z", "-")),
        ("LOGR", "DIST"),
    ],
    "logs": [False, False, False, True],
    "limits": [(-1, 3), (-1, 3), (-1, 3), (1e-3, CATALOG_RMAX),],
}
wide_r_settings = {
    "columns": [
        ("LOGR", "DIST"),
    ],
    "logs": [True,],
    "limits": [(1e-3, CATALOG_RMAX),],
}

columns = {
    "cols_dc": ["COLOR_G_R", "COLOR_R_I", "COLOR_I_Z",],
    "cols_wr": ["LOGR",],
    "cols_wcr": ["COLOR_G_R", "COLOR_R_I", "COLOR_I_Z", "LOGR",],
}

if __name__ == "__main__":

    mdl = pickle.load(open(wide_data_path, "rb"))
    deep = fio.read(deep_data_path)
    #
    master_seed = np.random.randint(0, np.iinfo(np.int32).max, 1)[0]
    rng = np.random.RandomState(seed=master_seed)
    seeds = rng.randint(0, np.iinfo(np.int32).max, 4)

    outname = root_path + tag + "_{:1d}".format(master_seed)
    print(outname)

    deep_c_settings = emulator.construct_deep_container(deep, deep_c_settings, seed=seeds[0])
    deep_smc_settings = emulator.construct_deep_container(deep, deep_smc_settings, seed=seeds[1])
    wide_cr_settings = emulator.construct_wide_container(mdl, wide_cr_settings, seed=seeds[2])
    wide_r_settings = emulator.construct_wide_container(mdl, wide_r_settings, seed=seeds[3])

    infodicts, samples = emulator.make_naive_infodicts(wide_cr_settings, wide_r_settings, deep_c_settings, deep_smc_settings,
                                                       columns, nsamples=NSAMPLES, nchunks=NCHUNKS, bandwidth=0.05, rmin=None,
                                                       rmax=np.log10(DRAW_RMAX))

    print("saving infodicts")
    fio.write(outname + "_samples.fits", samples.to_records(), clobber=True)
    for i, info in enumerate(infodicts):
        fname = outname + "_{:02}.p".format(i)
        pickle.dump(info, open(fname, "wb"))
    print("calculating scores")
    result = emulator.run_scores(infodicts)
    fio.write(outname + "_scores.fits", result.to_records(), clobber=True)