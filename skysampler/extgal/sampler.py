"""

Galsim extension package based on LOS constructors
"""

import galsim
import pickle
import ngmix

import numpy as np
import fitsio as fio


class SkySampler(object):
    _takes_rng = True
    _req_params = {"mock_file_list": str, "icl_file_list": str}
    _opt_params = {}
    _single_params = []
    def __init__(self, mock_file_list, icl_file_list, rng=None):

        with open(mock_file_list) as file:
            self.mock_file_list = file.readlines()

        self.itile = None
        self.mock = None
        self.ngal = None

        if icl_file_list is not None:
            with open(icl_file_list) as file:
                self.icl_file_list = file.readlines()
        else:
            self.icl_file_list = None

    def get_row(self, index):
        return self.mock[index]

    def get_nobj(self):
        return self.ngal

    def get_columns(self):
        return self.mock.dtype.names

    def read_mock(self):
        if self.itile < len(self.mock_file_list):
            print("reading table...")
            fname = self.mock_file_list[self.itile]
            print(fname)
            self.mock = fio.read(fname)
            self.ngal = len(self.mock)
            print("read", self.ngal, "objects...")
        else:
            raise IndexError("Ran out of tiles to render")

    def set_tile_num(self, num):
        self.itile = num

    def get_tile_num(self):
        return self.itile

    def safe_setup(self, itile):
        self.itile = itile
        if self.mock is None:
            self.read_mock()





def sky_row(config, base, name):
    index, index_key = galsim.config.GetIndex(config, base)
    ii = index - base["start_obj_num"]
    # print(ii, index)

    if base.get('_sky_sampler_index', None) != ii:
        sampler = galsim.config.GetInputObj('sky_sampler', config, base, name)
        sampler.safe_setup(base["tile_num"])

        base['_sky_row_data'] = sampler.get_row(ii)
        base['_sky_sampler_index'] = ii
        base['_sky_columns'] = sampler.get_columns()

    return base['_sky_row_data'], base['_sky_columns']


def sky_value(config, base, value_type):
    row, colnames = sky_row(config, base, value_type)
    col = galsim.config.ParseValue(config, 'col', base, str)[0]
    # print(col)
    if "FLUX" in col:
        col = "FLUX_" + str(base["band"]).upper()
    # TODO check this
    # if col == "SHEAR_G1" or col == "SHEAR_G2":
    #     res = 0.
    #     shear = base["shear_settings"]["value"]
    #     direction = base["shear_settings"]["direction"]
    #
        # if direction == "G1" and col == "SHEAR_G1":
        #     res = shear
        # elif direction == "G2" and col == "SHEAR_G2":
        #     res = shear
        # elif direction == "GT":
        #     phi = np.arctan2(row["Y"], row["X"])
        #     if col == "SHEAR_G1":
        #         res = -1. * shear * np.cos(2. * phi)
        #     elif col == "SHEAR_G2":
        #         res = -1. * shear * np.sin(2. * phi)
        # elif direction == "GX":
        #     phi = np.arctan2(row["Y"], row["X"])
        #     if col == "SHEAR_G1":
        #         res = 1. * shear * np.cos(2. * phi)
        #     elif col == "SHEAR_G2":
        #         res = -1. * shear * np.sin(2. * phi)

    # else:
    icol = colnames.index(col)
    res = float(row[icol])

    return res

def sky_tile_id(config, base, value_type):
    return base["tile_num"]

galsim.config.RegisterInputType('sky_sampler', galsim.config.InputLoader(SkySampler, file_scope=True))
galsim.config.RegisterValueType('sky_value', sky_value, [float], input_type='sky_sampler')
galsim.config.RegisterValueType('sky_tile_id', sky_tile_id, [int], input_type='sky_sampler')


def _next_bdf_obj(config, base, ignore, gsparams, logger):

    # Read next line from catalog
    index, index_key = galsim.config.GetIndex(config, base)
    ii = index - base["start_obj_num"]
    # print(ii)

    if base.get('_sky_sampler_index', None) != ii:
        sampler = galsim.config.GetInputObj('sky_sampler', config, base, 'sky_sampler')
        sampler.safe_setup(base["tile_num"])

        row = sampler.get_row(ii)
        cols = sampler.get_columns()
        base['_sky_row_data'] = row
        base['_sky_sampler_index'] = ii
        base['_sky_columns'] = cols

    row = base['_sky_row_data']

    bdf_pars = np.zeros(7)
    bdf_pars[2] = row["E1"]
    bdf_pars[3] = row["E2"]
    bdf_pars[4] = row["TSIZE"]
    bdf_pars[5] = row["FRACDEV"]

    bdf_pars[6] = row["FLUX_" + str(base["band"]).upper()]
    galmaker = ngmix.gmix.GMixBDF(bdf_pars)
    gs_profile = galmaker.make_galsim_object()

    return gs_profile, False


def _mock_bdf(config, base, ignore, gsparams, logger):

    print("mock_bdf")
    bdf_pars = np.zeros(7)
    bdf_pars[2] = 0.2
    bdf_pars[3] = 0.2
    bdf_pars[4] = 1.5
    bdf_pars[5] = 1.
    bdf_pars[6] = 2000.
    logger.info("Building GMixModel galaxy with bdf_pars: %s" % repr(bdf_pars))

    galmaker = ngmix.gmix.GMixBDF(bdf_pars)
    gs_profile = galmaker.make_galsim_object()

    return gs_profile, True


def _bdf_obj(config, base, ignore, gsparams, logger):
    # print(gsparams)
    # req = {"e1": float, "e2":float, "tsize":float, "fracdev":float, "flux": float}
    # print(config["e1"])
    # kwargs,safe = galsim.config.GetAllParams(config, base, req=req)

#     # Read next line from catalog
#     e1 = galsim.config.ParseValue(config, 'e1', base, str)[0]
#     e2 = galsim.config.ParseValue(config, 'e2', base, str)[0]
#     tsize = galsim.config.ParseValue(config, 'tsize', base, str)[0]
#     fracdev = galsim.config.ParseValue(config, 'fracdev', base, str)[0]
#     flux = galsim.config.ParseValue(config, 'flux', base, str)[0]
#

    # print(config["e1"])
    # raise KeyboardInterrupt
    bdf_pars = np.zeros(7)
    bdf_pars[2] = gsparams["e1"]
    bdf_pars[3] = gsparams["e2"]
    bdf_pars[4] = gsparams["tsize"]
    bdf_pars[5] = gsparams["fracdev"]
    bdf_pars[6] = gsparams["flux"]
    galmaker = ngmix.gmix.GMixBDF(bdf_pars)
    gs_profile = galmaker.make_galsim_object()
#
    # return gs_profile, False


galsim.config.RegisterObjectType('MockBDF', _mock_bdf, "MockBDF")
galsim.config.RegisterObjectType('BDFCat', _next_bdf_obj, "BDFCat")
galsim.config.RegisterObjectType('BDF', _bdf_obj, "BDF")
