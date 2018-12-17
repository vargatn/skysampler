"""

Galsim extension package based on LOS constructors
"""

import galsim
import pickle


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
    _req_params = {"file_name": str, "colnames": list}
    _opt_params = {}
    _single_params = []
    # colnames =
    def __init__(self, file_name, colnames, rng=None):
        self.mock = pickle.load(open(file_name, "rb"))
        self.colnames = list(self.mock.columns)
        self.ngal = len(self.mock)

    def get_row(self, index):
        return self.mock.iloc[index]

def SkyRow(config, base, name):
    index, index_key = galsim.config.GetIndex(config, base)
    # TODO actual catalog index is a combination of these
    # FIXME for this version we have to use 1 tile only
    ii = index - base["start_obj_num"]
    # tile should be choosen based on tile_num
    # i = index - start_obj_num  of tile

    # print("index_key", index_key)
    # print(base.keys())
    # print(base["start_obj_num"], base["tile_start_obj_num"], base["tile_start_obj_num"])
    # print(base["index_key"], base["band_num"], base["band"], base["tile_num"])

    if base.get('_sky_sampler_index',None) != ii:
        sampler = galsim.config.GetInputObj('sky_sampler', config, base, name)

        base['_sky_row_data'] = sampler.get_row(ii)
        base['_sky_sampler_index'] = ii
        base['_sky_colnames'] = sampler.colnames

    print(ii, base['_sky_row_data']["X"], base['_sky_row_data']["Y"])
    return base['_sky_row_data'], base['_sky_colnames']


def SkyValue(config, base, value_type):
    # sampler = galsim.config.GetInputObj('sky_sampler', config, base, value_type)
    # print(sampler)
    # print(sampler.mock)
    # row_data, colnames = SkyRow(config, base, value_type)
    # col = galsim.config.ParseValue(config, 'col', base, str)[0]
    # if "FLUX" in col:
    #     col = "FLUX_" + str(base["band"]).upper()
    # print(col)
    # res = float( row_data[ colnames.index(col) ] )
    # return res
    return 1.

def SkyNum(config, base, value_type):
    fname = base["input"]["sky_sampler"]["file_name"]
    sampler = SkySampler(fname, None)
    # print(sampler.ngal)
    # return sampler.ngal
    return 10

galsim.config.RegisterInputType('sky_sampler', galsim.config.InputLoader(SkySampler))
galsim.config.RegisterValueType('sky_value', SkyValue, [float], input_type='sky_sampler')
galsim.config.RegisterValueType('sky_num', SkyNum, [int], input_type="sky_sampler")


