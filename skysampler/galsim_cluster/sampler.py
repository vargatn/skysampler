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
    _req_params = {"mock_file_list": str,}
    _opt_params = {}
    _single_params = []
    def __init__(self, mock_file_list, rng=None):
        """
        What this should actually read is a file list,
        and loop through them as the tiles increase

        Parameters
        ----------
        file_name
        rng
        """
        # print("HERE WE ARE STARTING")
        with open(mock_file_list) as file:
            self.mock_file_list = file.readlines()
        self.read_mock(0)


    def get_row(self, index):
        return self.mock[index]

    def get_nobj(self):
        return self.ngal

    def get_columns(self):
        return self.mock.dtype.names

    # TODO add repeat key to this...
    def read_mock(self, itile):
        if itile < len(self.mock_file_list):
            self.mock = fio.read(self.mock_file_list[itile])
            self.itile = itile
            self.ngal = len(self.mock)
        else:
            raise IndexError("Ran out of tiles to render")

    def set_tile_num(self, num):
        self.itile = num
        self.read_mock(self.itile)

    def get_tile_num(self):
        return self.itile

def sky_row(config, base, name):
    # TODO actual catalog index is a combination of these
    # FIXME for this version we have to use 1 tile only
    index, index_key = galsim.config.GetIndex(config, base)
    ii = index - base["start_obj_num"]
    # TODO maybe use index_key = obj_in_file instead of this
    # print(ii)
    # print(itile)
    # print("config.tile_num", config["tile_num"])

    if base.get('_sky_sampler_index', None) != ii:
        sampler = galsim.config.GetInputObj('sky_sampler', config, base, name)

        base['_sky_row_data'] = sampler.get_row(ii)
        base['_sky_sampler_index'] = ii
        base['_sky_columns'] = sampler.get_columns()

    return base['_sky_row_data'], base['_sky_columns']


def sky_value(config, base, value_type):
    # print("+++++++++++", base["tile_num"], base["image_num"])

    # row, colnames = SkyRow(config, base, value_type)
    # col = galsim.config.ParseValue(config, 'col', base, str)[0]
    # if "FLUX" in col:
    #     col = "FLUX_" + str(base["band"]).upper()
    # icol = colnames.index(col)
    # res = float(row[icol])
    return 1.

def sky_num(config, base, value_type):
    # print(">>>>>>>>>>>", base["tile_num"], base["image_num"])

    # sky = galsim.config.GetInputObj("sky_sampler", config, base, 'SkyNum')
    # print("itile", sky.get_itile)

    # index, index_key = galsim.config.GetIndex(config, base)
    # print("HERE............")
    # print(config)
    # print("END.............")
    # print(base.keys())
    # print(base["tile_num"])

    val = 18
    return val


# galsim.config.RegisterInputType('coadd', galsim.config.InputLoader(CoaddInput, file_scope=True))

galsim.config.RegisterInputType('sky_sampler', galsim.config.InputLoader(SkySampler, file_scope=True))
galsim.config.RegisterValueType('sky_value', sky_value, [float], input_type='sky_sampler')
galsim.config.RegisterValueType('sky_num', sky_num, [float], input_type="sky_sampler")

# def sky_tile_id(config, base, value_type):
#     index, index_key = galsim.config.GetIndex(config, base)
#     print("SKY_TILE_ID", index)
#     return 11

# galsim.config.RegisterValueType("sky_tile_id", sky_tile_id, [int], input_type="sky_sampler")

# galsim.config.RegisterValueType('ThisTileNum', ThisTileNum, [int], input_type='sky_sampler')

# galsim.config.image_scattered

