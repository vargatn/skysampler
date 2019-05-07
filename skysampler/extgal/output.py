from __future__ import print_function
import galsim
import os
import numpy as np
import copy
import fitsio as fio

from cluster_pipeline.multiband_fits import MultibandFitsBuilder

from galsim.config.output import OutputBuilder


def extract_nobjects(config, base, ntiles):
    """
    This should read the number of objects from an input catalog file list...

    Should be really lightweight and not do much data processing

    Returns
    -------

    """

    mock_file_path = base["input"]["sky_sampler"]["mock_file_list"]
    with open(mock_file_path) as file:
        mock_filenames = file.readlines()

    nobjs = []
    for filename in mock_filenames[:ntiles]:
        tab = fio.FITS(filename)[1]
        nobjs.append(tab.get_nrows())
    # print("NOBJS:", nobjs)
    return nobjs
    # return [999, 999]


class CatalogOutputBuilder(MultibandFitsBuilder):
    """
    The philosophy of this class is to provide output for a very specific type of Image

    We take everything from a predefined catalog, and draw them to images
    """
    def setup(self, config, base, file_num, logger):

        keys = ['tile_num', 'band_num']
        diff = set(keys).difference(set(galsim.config.process.valid_index_keys))
        galsim.config.valid_index_keys += sorted(diff)

        keys = ['tile_num', 'band_num', 'tile_start_obj_num', 'band']
        diff = set(keys).difference(set(galsim.config.eval_base_variables))
        galsim.config.eval_base_variables += diff

        if 'ntiles' in config:
            try:
                ntiles = galsim.config.ParseValue(config, 'ntiles', base, int)[0]
            except:
                galsim.config.ProcessInput(base, safe_only=True)
                ntiles = galsim.config.ParseValue(config, 'ntiles', base, int)[0]
        else:
            ntiles = 1

        bands = config["bands"]
        nbands = len(config["bands"])
        tile_num = file_num // nbands
        band_num = file_num % nbands

        base['tile_num'] = tile_num
        base['band_num'] = band_num
        base['band'] = config['bands'][band_num]

        nobj_list = extract_nobjects(config, base, ntiles)
        # nobj_list = [10, ] * ntiles
        base['image']['nobjects'] = {
            'type': 'List',
            'items': nobj_list,
            'index_key': 'tile_num',
            '_setup_as_list': True,
        }
        OutputBuilder.setup(self, config, base, file_num, logger)


galsim.config.output.RegisterOutputType('MultibandCatalogFits',  CatalogOutputBuilder())

