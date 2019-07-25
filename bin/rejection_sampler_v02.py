"""
Rejection sampler draw
"""

import numpy as np
import pandas as pd
import pickle
import fitsio as fio
import skysampler.emulator as emulator
import skysampler.utils as utils


VERSION = "02-gr-ri-naive"

root_path = "/e/eser2/vargatn/EMULATOR/GAMMA/resamples/"
wide_data_path = "/e/eser2/vargatn/EMULATOR/GAMMA/multi-indexer-gamma_v001_clust__z0_l1.p"
deep_data_path = "/e/eser2/vargatn/DES/SIM_DATA/run-vd09-SN-C3_trim_extcorr.fits"
RMAX = 20.

NSAMPLES = 1e6
NCHUNKS = 200
NPROCESS = 30

deep_c_settings = {
    "columns": [
        ("COLOR_G_R", (("bdf_mag", 0), ("bdf_mag", 1), "-")),
    ],
    "logs": [False,],
    "limits": [(-2, 4),],
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
    "fname": deep_data_path,
}
wide_cr_settings = {
    "columns": [
        ("COLOR_G_R", ("MOF_CM_MAG_CORRECTED_G", "MOF_CM_MAG_CORRECTED_R", "-")),
        ("LOGR", "DIST"),
    ],
    "logs": [False, True],
    "limits": [(-2, 4), (1e-3, RMAX),],
    "fname": wide_data_path,
}
wide_r_settings = {
    "columns": [
        ("LOGR", "DIST"),
    ],
    "logs": [True,],
    "limits": [(1e-3, RMAX),],
    "fname": wide_data_path,
}

wide_smcr_settings = {
    "columns": [
        ("GABS", ("MOF_CM_G_1", "MOF_CM_G_1", "SQSUM")),
        ("SIZE", ("MOF_CM_T", 1, "+")),
        ("FRACDEV", "MOF_CM_FRACDEV"),
        ("MAG_I", "MOF_CM_MAG_CORRECTED_I"),
        ("COLOR_G_R", ("MOF_CM_MAG_CORRECTED_G", "MOF_CM_MAG_CORRECTED_R", "-")),
        ("COLOR_R_I", ("MOF_CM_MAG_CORRECTED_R", "MOF_CM_MAG_CORRECTED_I", "-")),
        ("COLOR_I_Z", ("MOF_CM_MAG_CORRECTED_I", "MOF_CM_MAG_CORRECTED_Z", "-")),
        ("LOGR", "DIST"),
    ],
    "logs": [False, True, False, False, False, False, False, True],
    "limits": [(0, 1), (-1, 5), (-3, 4), (16, 26), (-2, 4), (-2, 4), (-2, 4), (1e-3, RMAX)],
}

prior_cols = {
    "cols_dc": ["COLOR_G_R",],
    "cols_wr": ["LOGR",],
    "cols_wcr": ["COLOR_G_R", "LOGR",],
}


if __name__ == "__main__":
    mdl = pickle.load(open(wide_data_path, "rb"))
    wide_smcr_settings = emulator.construct_wide_container(mdl, wide_smcr_settings)
    wide_cr_settings = emulator.construct_wide_container(mdl, wide_cr_settings)
    wide_r_settings = emulator.construct_wide_container(mdl, wide_r_settings)

    deep_smc_settings = emulator.construct_deep_container(deep_data_path, deep_smc_settings)
    deep_c_settings = emulator.construct_deep_container(deep_data_path, deep_c_settings)

    columns = [
        ("GABS", deep_smc_settings),
        ("SIZE", deep_smc_settings),
        ("FRACDEV", deep_smc_settings),
        ("MAG_I", deep_smc_settings),
        ("COLOR_G_R", deep_smc_settings),
        ("COLOR_R_I", deep_smc_settings),
        ("COLOR_I_Z", deep_smc_settings),
        ("LOGR", wide_r_settings),
    ]

    means = []
    sigmas = []
    cols = []
    for (col, stt) in columns:
        cols.append(col)
        _cols = np.array(list(stt["container"].data.columns))
        icol = np.where(col == _cols)[0][0]
        #     print(col, icol, stt["container"].mean[icol])
        means.append(stt["container"].mean[icol])
        sigmas.append(stt["container"].sigma[icol])

    means = np.array(means)
    means = pd.DataFrame(data=means[:, np.newaxis].T, columns=cols)

    sigmas = np.array(sigmas)
    sigmas = pd.DataFrame(data=sigmas[:, np.newaxis].T, columns=cols)

    cont = wide_smcr_settings["container"]
    wide_smcr_cont = emulator.DualContainer(columns=cont.columns,
                                            mean=means[list(cont.columns)].values,
                                            sigma=sigmas[list(cont.columns)].values)
    wide_smcr_cont.set_data(cont.data, weights=cont.weights)

    cont = wide_cr_settings["container"]
    wide_cr_cont = emulator.DualContainer(columns=cont.columns,
                                          mean=means[list(cont.columns)].values,
                                          sigma=sigmas[list(cont.columns)].values)
    wide_cr_cont.set_data(cont.data, weights=cont.weights)

    cont = wide_r_settings["container"]
    wide_r_cont = emulator.DualContainer(columns=cont.columns,
                                         mean=means[list(cont.columns)].values,
                                         sigma=sigmas[list(cont.columns)].values)
    wide_r_cont.set_data(cont.data, weights=cont.weights)

    cont = deep_smc_settings["container"]
    deep_smc_cont = emulator.DualContainer(columns=cont.columns,
                                           mean=means[list(cont.columns)].values,
                                           sigma=sigmas[list(cont.columns)].values)
    deep_smc_cont.set_data(cont.data, weights=cont.weights)

    cont = deep_c_settings["container"]
    deep_c_cont = emulator.DualContainer(columns=cont.columns,
                                         mean=means[list(cont.columns)].values,
                                         sigma=sigmas[list(cont.columns)].values)
    deep_c_cont.set_data(cont.data, weights=cont.weights)

    sample, infodicts = emulator.make_naive_infodicts(wide_cr_cont,
                                                      wide_r_cont,
                                                      deep_c_cont,
                                                      deep_smc_cont,
                                                      nsamples=NSAMPLES, cols=prior_cols,
                                                      nchunks=NCHUNKS, rmax=5, bandwidth=0.1)
    fname = root_path + "resample_" + VERSION + ".p"
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
        fname = root_path + "resample_" + VERSION + "_chunk_{:02d}.p".format(i)
        print(fname)
        pickle.dump(res, open(fname, "wb"))
