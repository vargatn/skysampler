"""

Galsim extension package based on LOS constructors
"""

import galsim
import pickle
import numpy as np
import fitsio as fio


class SkyCreator(object):
    """
    Creates N realizations in File
    """
    def __init__(self):
        pass


class SkySampler(object):
    """
    This is the class which gets exposed to galsim

    Everything goes in here


    Should be able
    """
    _takes_rng = True
    _req_params = {"file_name": str,}
    _opt_params = {}
    _single_params = []
    def __init__(self, file_name, rng=None):
        self.mock = fio.read(file_name)
        self.ngal = len(self.mock)

    def get_row(self, index):
        return self.mock[index]

    def get_nobj(self):
        return self.ngal

    def get_columns(self):
        return self.mock.dtype.names

def SkyRow(config, base, name):
    # TODO actual catalog index is a combination of these
    # FIXME for this version we have to use 1 tile only
    index, index_key = galsim.config.GetIndex(config, base)
    ii = index - base["start_obj_num"]
    print(ii)

    if base.get('_sky_sampler_index', None) != ii:
        sampler = galsim.config.GetInputObj('sky_sampler', config, base, name)

        base['_sky_row_data'] = sampler.get_row(ii)
        base['_sky_sampler_index'] = ii
        base['_sky_columns'] = sampler.get_columns()

    return base['_sky_row_data'], base['_sky_columns']


def SkyValue(config, base, value_type):
    row, colnames = SkyRow(config, base, value_type)
    col = galsim.config.ParseValue(config, 'col', base, str)[0]
    if "FLUX" in col:
        col = "FLUX_" + str(base["band"]).upper()
    icol = colnames.index(col)
    res = float(row[icol])
    if col == "ANGLE":
        res *= galsim.degrees
    return res

def SkyNum(config, base, value_type):
    print("here")
    # print(base[])
    val = len(fio.read(base["input"]["sky_sampler"]["file_name"]))
    # sampler = galsim.config.GetInputObj('sky_sampler', config, base, value_type)
    # val = sampler.get_nobj()
    print(val)
    return val


galsim.config.RegisterInputType('sky_sampler', galsim.config.InputLoader(SkySampler))
galsim.config.RegisterValueType('sky_value', SkyValue, [float], input_type='sky_sampler')
galsim.config.RegisterValueType('sky_num', SkyNum, [int], input_type="sky_sampler")


