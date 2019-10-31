"""
Rejection sampler draw
"""

import numpy as np
import pickle
import fitsio as fio
import skysampler.old_emulator as emulator
import skysampler.utils as utils

VERSION = "01-gr-ri"


wide_data_path = "/e/eser2/vargatn/EMULATOR/GAMMA/multi-indexer-gamma_v001_clust__z0_l1.p"
deep_data_path = "/e/eser2/vargatn/DES/SIM_DATA/run-vd09-SN-C3_trim_extcorr.fits"
RMAX = 5.

NSAMPLES = 2e6
NCHUNKS = 400
NPROCESS = 30

deep_c_settings = {
    "columns": [
        ("COLOR_G_R", (("bdf_mag", 0), ("bdf_mag", 1), "-")),
        ("COLOR_R_I", (("bdf_mag", 1), ("bdf_mag", 2), "-")),
    ],
    "logs": [False, False],
    "limits": [(-2, 4), (-2, 4)],
    "emulator": {
        "bandwidth": 0.1
    },
    "fname": deep_data_path,
}
deep_smc_settings = {
    "columns": [
        ("GABS", (("bdf_g", 0), ("bdf_g", 1), "SQSUM")),
        ("SIZE", ("bdf_T", 1, "+")),
        ("FRACDEV", "bdf_fracdev"),
        ("MAG_I", ("bdf_mag", 2)),
        ("COLOR_G_R", (("bdf_mag", 0), ("bdf_mag", 1), "-")),
        ("COLOR_R_I", (("bdf_mag", 1), ("bdf_mag", 2), "-")),
        ("COLOR_I_Z", (("bdf_mag", 2), ("bdf_mag", 3), "-")),
    ],
    "logs": [False, True, False, False, False, False, False],
    "limits": [(0, 1), (-1, 5), (-3, 4), (16, 26), (-2, 4), (-2, 4), (-2, 4)],
    "emulator": {
        "bandwidth": 0.15,
        "ref_axes": 3,
        "nslices": 5,
        "tomographic_weights": ((0, 1, 1, 0, 0),),
        "nbins": 100,
        "eta": 1. * np.array([1., 0., 0., 0., 1., 0., 1.]),
        "window_size": 15,
    },
    "fname": deep_data_path,
}
wide_cr_settings = {
    "columns": [
        ("COLOR_G_R", ("MOF_CM_MAG_CORRECTED_G", "MOF_CM_MAG_CORRECTED_R", "-")),
        ("COLOR_R_I", ("MOF_CM_MAG_CORRECTED_R", "MOF_CM_MAG_CORRECTED_I", "-")),
        ("LOGR", "DIST"),
    ],
    "logs": [False, False, True],
    "limits": [(-2, 4), (-2, 4), (1e-3, RMAX),],
    "emulator": {
        "bandwidth": 0.1,
        "ref_axes": 2,
        "nslices": 7,
        "tomographic_weights": ((0, 1, 0, 0, 0, 0, 0),),
        "nbins": 100,
        "eta": 1 * np.array([0, 2, 1]),
        "window_size": 10,
    },
    "fname": wide_data_path,
}
wide_r_settings = {
    "columns": [
        ("LOGR", "DIST"),
    ],
    "logs": [True,],
    "limits": [(1e-3, RMAX),],
    "emulator": {
        "bandwidth": 0.1,
        "nbins": 100,
        "eta": 0.1,
        "window_size": 15,
    },
    "fname": wide_data_path,
}

prior_cols = {
    "cols_dc": ["COLOR_G_R", "COLOR_R_I"],
    "cols_wr": ["LOGR",],
    "cols_wcr": ["COLOR_G_R", "COLOR_R_I", "LOGR",],
}


if __name__ == "__main__":
    mdl = pickle.load(open(wide_data_path, "rb"))
    wide_cr_settings = emulator.construct_wide_container(mdl, wide_cr_settings)
    wide_r_settings = emulator.construct_wide_container(mdl, wide_r_settings)

    deep_smc_settings = emulator.construct_deep_container(deep_data_path, deep_smc_settings)
    deep_c_settings = emulator.construct_deep_container(deep_data_path, deep_c_settings)

    sample, infodicts = emulator.make_infodicts(wide_cr_settings,
                                                wide_r_settings,
                                                deep_c_settings,
                                                deep_smc_settings,
                                                nsamples=NSAMPLES, cols=prior_cols,
                                                nchunks=NCHUNKS)

    fname = "/e/eser2/vargatn/EMULATOR/GAMMA/resamples/resample_" + VERSION + ".p"
    res = {
        "sample": sample,
        "infodicts": infodicts,
    }
    pickle.dump(res, open(fname, "wb"))
    print(fname)

    chunked_infodicts = utils.partition(infodicts, int(np.ceil(NCHUNKS / NPROCESS)))
    for i, chunk in enumerate(chunked_infodicts):
        result = emulator.run_scores(chunk)
        res = {
            "result": result,
        }
        fname = "resample_" + VERSION + "_chunk_{:02d}.p".format(i)
        print(fname)
        pickle.dump(res, open(fname, "wb"))
