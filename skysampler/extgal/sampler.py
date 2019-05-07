"""

Galsim extension package based on LOS constructors
"""

import galsim
import pickle
import numpy as np
import fitsio as fio


class SkySampler(object):
    _takes_rng = True
    _req_params = {"mock_file_list": str,}
    _opt_params = {}
    _single_params = []
    def __init__(self, mock_file_list, rng=None):

        with open(mock_file_list) as file:
            self.mock_file_list = file.readlines()

        self.itile = None
        self.mock = None
        self.ngal = None

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
    if "FLUX" in col:
        col = "FLUX_" + str(base["band"]).upper()
    # TODO check this
    if col == "SHEAR_G1" or col == "SHEAR_G2":
        res = 0.
        shear = base["shear_settings"]["value"]
        direction = base["shear_settings"]["direction"]

        if direction == "G1" and col == "SHEAR_G1":
            res = shear
        elif direction == "G2" and col == "SHEAR_G2":
            res = shear
        elif direction == "GT":
            phi = np.arctan2(row["Y"], row["X"])
            if col == "SHEAR_G1":
                res = -1. * shear * np.cos(2. * phi)
            elif col == "SHEAR_G2":
                res = -1. * shear * np.sin(2. * phi)
        elif direction == "GX":
            phi = np.arctan2(row["Y"], row["X"])
            if col == "SHEAR_G1":
                res = 1. * shear * np.cos(2. * phi)
            elif col == "SHEAR_G2":
                res = -1. * shear * np.sin(2. * phi)

    else:
        icol = colnames.index(col)
        res = float(row[icol])

    return res

def sky_tile_id(config, base, value_type):
    return base["tile_num"]

galsim.config.RegisterInputType('sky_sampler', galsim.config.InputLoader(SkySampler, file_scope=True))
galsim.config.RegisterValueType('sky_value', sky_value, [float], input_type='sky_sampler')
galsim.config.RegisterValueType('sky_tile_id', sky_tile_id, [int], input_type='sky_sampler')
