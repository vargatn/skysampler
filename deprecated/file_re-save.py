from __future__ import print_function, division
import fitsio as fio
import numpy as np
import os
import pandas as pd

import skysampler.emulator as emulator

try:
    import cPickle as pickle
except:
    import pickle

LOGR_DRAW_RMINS = [-3, -0.5, 0., 0.5]
LOGR_DRAW_RMAXS = [-0.5, 0., 0.5, 1.2]


wide_cr_settings = {
    "columns": [
        ("MAG_I", "MOF_CM_MAG_CORRECTED_I"),
        ("COLOR_G_R", ("MOF_CM_MAG_CORRECTED_G", "MOF_CM_MAG_CORRECTED_R", "-")),
        ("COLOR_R_I", ("MOF_CM_MAG_CORRECTED_R", "MOF_CM_MAG_CORRECTED_I", "-")),
        ("COLOR_I_Z", ("MOF_CM_MAG_CORRECTED_I", "MOF_CM_MAG_CORRECTED_Z", "-")),
        ("LOGR", "DIST"),
    ],
    "logs": [False, False, False, False, True],
    "limits": [(17, 22.5), (-1, 3), (-1, 3), (-1, 3), (1e-3, 50.), ],
}

zls = [(0,0), (1, 0), (2, 0)]
if __name__ == "__main__":

    for zl in zls:
        z = zl[0]
        l = zl[1]
        wide_data_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_clust_z" + str(
            z) + "_l" + str(l) + ".p"
        print(wide_data_path)
        mdl = pickle.load(open(wide_data_path, "rb"))

        tmp_wide_cr_settings = wide_cr_settings.copy()
        # tmp_wide_cr_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
        _wide_cr_settings = emulator.construct_wide_container(mdl, tmp_wide_cr_settings, seed=5)

        cont_wcr = _wide_cr_settings["container"]
        wide_data_path_out = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_clust_z" + str(
            z) + "_l" + str(l) + ".h5"
        print(wide_data_path_out)
        tab = pd.DataFrame()
        for col in cont_wcr.columns:
            tab[col] = cont_wcr.data[col]
            tab["WEIGHTS"] = cont_wcr.weights

        tab.to_hdf(wide_data_path_out, key="data")


        wide_data_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_rands_z" + str(
            z) + "_l" + str(l) + ".p"
        print(wide_data_path)

        mdl = pickle.load(open(wide_data_path, "rb"))

        tmp_wide_cr_settings = wide_cr_settings.copy()
        # tmp_wide_cr_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
        _wide_cr_settings = emulator.construct_wide_container(mdl, tmp_wide_cr_settings, seed=5)

        cont_wcr = _wide_cr_settings["container"]
        wide_data_path_out = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_rands_z" + str(
            z) + "_l" + str(l) + ".h5"
        print(wide_data_path_out)
        tab = pd.DataFrame()
        for col in cont_wcr.columns:
            tab[col] = cont_wcr.data[col]
            tab["WEIGHTS"] = cont_wcr.weights

        tab.to_hdf(wide_data_path_out, key="data")


    # l = 1
    # z = 0
    # wide_data_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_clust_z" + str(
    #     z) + "_l" + str(l) + ".p"
    # mdl = pickle.load(open(wide_data_path, "rb"))
    #
    # tmp_wide_cr_settings = wide_cr_settings.copy()
    # # tmp_wide_cr_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
    # _wide_cr_settings = emulator.construct_wide_container(mdl, tmp_wide_cr_settings, seed=5)
    #
    # cont_wcr = _wide_cr_settings["container"]
    # wide_data_path_out = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_clust_z" + str(
    #     z) + "_l" + str(l) + ".h5"
    # print(wide_data_path_out)
    # tab = pd.DataFrame()
    # for col in cont_wcr.columns:
    #     tab[col] = cont_wcr.data[col]
    #     tab["WEIGHTS"] = cont_wcr.weights
    #
    # tab.to_hdf(wide_data_path_out, key="data")
    #
    # l = 1
    # z = 1
    # wide_data_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_clust_z" + str(
    #     z) + "_l" + str(l) + ".p"
    # mdl = pickle.load(open(wide_data_path, "rb"))
    #
    # tmp_wide_cr_settings = wide_cr_settings.copy()
    # # tmp_wide_cr_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
    # _wide_cr_settings = emulator.construct_wide_container(mdl, tmp_wide_cr_settings, seed=5)
    #
    # cont_wcr = _wide_cr_settings["container"]
    # wide_data_path_out = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_clust_z" + str(
    #     z) + "_l" + str(l) + ".h5"
    # print(wide_data_path_out)
    # tab = pd.DataFrame()
    # for col in cont_wcr.columns:
    #     tab[col] = cont_wcr.data[col]
    #     tab["WEIGHTS"] = cont_wcr.weights
    #
    # tab.to_hdf(wide_data_path_out, key="data")
    #
    # l = 1
    # z = 2
    # wide_data_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_clust_z" + str(
    #     z) + "_l" + str(l) + ".p"
    # mdl = pickle.load(open(wide_data_path, "rb"))
    #
    # tmp_wide_cr_settings = wide_cr_settings.copy()
    # # tmp_wide_cr_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
    # _wide_cr_settings = emulator.construct_wide_container(mdl, tmp_wide_cr_settings, seed=5)
    #
    # cont_wcr = _wide_cr_settings["container"]
    # wide_data_path_out = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_clust_z" + str(
    #     z) + "_l" + str(l) + ".h5"
    # print(wide_data_path_out)
    # tab = pd.DataFrame()
    # for col in cont_wcr.columns:
    #     tab[col] = cont_wcr.data[col]
    #     tab["WEIGHTS"] = cont_wcr.weights
    #
    # tab.to_hdf(wide_data_path_out, key="data")
    #
    # l = 1
    # z = 0
    # wide_data_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_rands_z" + str(
    #     z) + "_l" + str(l) + ".p"
    # mdl = pickle.load(open(wide_data_path, "rb"))
    #
    # tmp_wide_cr_settings = wide_cr_settings.copy()
    # # tmp_wide_cr_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
    # _wide_cr_settings = emulator.construct_wide_container(mdl, tmp_wide_cr_settings, seed=5)
    #
    # cont_wcr = _wide_cr_settings["container"]
    # wide_data_path_out = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_rands_z" + str(
    #     z) + "_l" + str(l) + ".h5"
    # print(wide_data_path_out)
    # tab = pd.DataFrame()
    # for col in cont_wcr.columns:
    #     tab[col] = cont_wcr.data[col]
    #     tab["WEIGHTS"] = cont_wcr.weights
    #
    # tab.to_hdf(wide_data_path_out, key="data")

    # l = 1
    # z = 1
    # wide_data_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_rands_z" + str(
    #     z) + "_l" + str(l) + ".p"
    # mdl = pickle.load(open(wide_data_path, "rb"))
    #
    # tmp_wide_cr_settings = wide_cr_settings.copy()
    # # tmp_wide_cr_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
    # _wide_cr_settings = emulator.construct_wide_container(mdl, tmp_wide_cr_settings, seed=5)
    #
    # cont_wcr = _wide_cr_settings["container"]
    # wide_data_path_out = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_rands_z" + str(
    #     z) + "_l" + str(l) + ".h5"
    # print(wide_data_path_out)
    # tab = pd.DataFrame()
    # for col in cont_wcr.columns:
    #     tab[col] = cont_wcr.data[col]
    #     tab["WEIGHTS"] = cont_wcr.weights
    #
    # tab.to_hdf(wide_data_path_out, key="data")
    #
    # l = 1
    # z = 2
    # wide_data_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_rands_z" + str(
    #     z) + "_l" + str(l) + ".p"
    # mdl = pickle.load(open(wide_data_path, "rb"))
    #
    # tmp_wide_cr_settings = wide_cr_settings.copy()
    # # tmp_wide_cr_settings["limits"][-1] = (10**-3, 10**LOGR_CAT_RMAXS[i])
    # _wide_cr_settings = emulator.construct_wide_container(mdl, tmp_wide_cr_settings, seed=5)
    #
    # cont_wcr = _wide_cr_settings["container"]
    # wide_data_path_out = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/multi-indexer-epsilon_narrow-z_zoom_high-l_v004/multi-indexer-epsilon_narrow-z_zoom_high-l_v004_rands_z" + str(
    #     z) + "_l" + str(l) + ".h5"
    # print(wide_data_path_out)
    # tab = pd.DataFrame()
    # for col in cont_wcr.columns:
    #     tab[col] = cont_wcr.data[col]
    #     tab["WEIGHTS"] = cont_wcr.weights
    #
    # tab.to_hdf(wide_data_path_out, key="data")