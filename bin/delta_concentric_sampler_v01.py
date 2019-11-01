from __future__ import print_function, division
import fitsio as fio
import numpy as np

import skysampler.emulator as emulator

try:
    import cPickle as pickle
except:
    import pickle

tag = "delta_concentric_sample_v01"
NSAMPLES = 100
NCHUNKS = 32
BANDWIDTH = 0.05
LOGR_DRAW_RMINS = [-3, -0.5, 0., 0.5]
LOGR_DRAW_RMAXS = [-0.5, 0., 0.5, 1.2]
LOGR_CAT_RMAXS = [0., 0.5, 1.2, 2.]

root_path = "/e/eser2/vargatn/EMULATOR/DELTA/resamples/"
wide_data_path = "/e/eser2/vargatn/EMULATOR/GAMMA/multi-indexer-gamma_v001_clust__z0_l1_py2.p"
deep_data_path = "/e/eser2/vargatn/EMULATOR/DELTA/run-ugriz-mof02_naive-cleaned.fits"

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
    "limits": [(-1, 3), (-1, 3), (-1, 3), (1e-3, 50.),],
}
wide_r_settings = {
    "columns": [
        ("LOGR", "DIST"),
    ],
    "logs": [True,],
    "limits": [(1e-3, 50.),],
}

columns = {
    "cols_dc": ["COLOR_G_R", "COLOR_R_I", "COLOR_I_Z",],
    "cols_wr": ["LOGR",],
    "cols_wcr": ["COLOR_G_R", "COLOR_R_I", "COLOR_I_Z", "LOGR",],
}

if __name__ == "__main__":

    nrbins = len(LOGR_DRAW_RMINS)
    print("started reading")
    mdl = pickle.load(open(wide_data_path, "rb"))
    deep = fio.read(deep_data_path)
    print("finished reading")

    master_seed = np.random.randint(0, np.iinfo(np.int32).max, 1)[0]
    rng = np.random.RandomState(seed=master_seed)
    seeds = rng.randint(0, np.iinfo(np.int32).max, nrbins * 4)

    print("starting concentric shell resampling")
    for i in np.arange(nrbins):
        print("rbin", i)

        outname = root_path + tag + "_{:1d}".format(master_seed) + "_rbin{:d}".format(i)
        print(outname)

        # update configs
        _deep_c_settings = emulator.construct_deep_container(deep, deep_c_settings, seed=seeds[nrbins * i + 0])
        _deep_smc_settings = emulator.construct_deep_container(deep, deep_smc_settings, seed=seeds[nrbins * i + 1])

        _wide_cr_settings = wide_cr_settings.copy()
        _wide_cr_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
        _wide_cr_settings = emulator.construct_wide_container(mdl, _wide_cr_settings, seed=seeds[nrbins * i + 2])

        _wide_r_settings = wide_r_settings.copy()
        _wide_r_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
        _wide_r_settings = emulator.construct_wide_container(mdl, _wide_r_settings, seed=seeds[nrbins * i + 3])

        # create infodicts
        infodicts, samples = emulator.make_naive_infodicts(_wide_cr_settings, _wide_r_settings, _deep_c_settings,
                                                           _deep_smc_settings,
                                                           columns, nsamples=NSAMPLES, nchunks=NCHUNKS, bandwidth=BANDWIDTH,
                                                           rmin=LOGR_DRAW_RMINS[i],
                                                           rmax=LOGR_DRAW_RMAXS[i])

        fname = outname + "_samples.fits"
        print(fname)
        fio.write(fname, samples.to_records(), clobber=True)
        master_dict = {
            "columns": infodicts[0]["columns"],
            "bandwidth": infodicts[0]["bandwidth"],
            "deep_c_settings": _deep_c_settings,
            "deep_smc_settings": _deep_smc_settings,
            "wide_r_settings": _wide_r_settings,
            "wide_cr_settings": _wide_cr_settings,
            "rmin": infodicts[0]["rmin"],
            "rmax": infodicts[0]["rmin"],
        }
        pickle.dump(master_dict, open(outname + ".p", "wb"))
        print("calculating scores")
        result = emulator.run_scores(infodicts)
        print("finished calculating scores")
        fname = outname + "_scores.fits"
        print(fname)
        fio.write(fname, result.to_records(), clobber=True)


