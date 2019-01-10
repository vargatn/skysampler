from __future__ import print_function
import galsim
import os
import numpy as np
import copy

from galsim.config.output import OutputBuilder


def extract_nobjects():
    """
    This should read the number of objects from an input catalog file list...

    Should be really lightweight and not do much data processing

    Returns
    -------

    """
    pass

class CatalogOutputBuilder(OutputBuilder):
    """
    The philosophy of this class is to provide output for a very specific type of Image

    We take everything from a predefined catalog, and draw them to images
    """
    def setup(self, config, base, file_num, logger):
        # TODO This should be super light-weight!
        # TODO sometimes this is called just stand-alon, sometimes before building a particular table

        print(galsim.config.process.valid_index_keys)
        print(galsim.config.eval_base_variables)

        print(">>>>>>>>>>>>>>>>>>")
        print("SETUP IS CALLED", file_num)

        # FIXME this is a bit broken here below...
        if 'tile_num' not in galsim.config.process.valid_index_keys:
            galsim.config.valid_index_keys += ['tile_num', 'band_num']
            galsim.config.eval_base_variables += ['tile_num', 'band_num', 'tile_start_obj_num', 'band']

        if 'ntiles' in config:
            try:
                ntiles = galsim.config.ParseValue(config, 'ntiles', base, int)[0]
            except:
                galsim.config.ProcessInput(base, safe_only=True)
                ntiles = galsim.config.ParseValue(config, 'ntiles', base, int)[0]
        else:
            ntiles = 1

        #
        # # TODO what we should do here is a clause to check if we can
        # self.sampler = galsim.config.GetInputObj('sky_sampler', config, base, 'sky_sampler')
        # print("SAMPLER", self.sampler)
        # Here we should also initialize the SAMPLER
        #we check how many tiles there are, and setup the sampler accordingly


        # TODO This part is not needed as we know the number of objects for each band
        # We'll be setting the random number seed to repeat for each band, which requires
        # querying the number of objects in the exposure.  This however leads to a logical
        # infinite loop if the number of objects is a random variate.  So to make this work,
        # we first get the number of objects in each exposure using a well-defined rng, and
        # save those values to a list, which is then fully deterministic for all other uses.

        if 'nobjects' not in base['image']:
            raise ValueError("image.nobjects is required for output type 'MultibandFitsBuilder'")
        nobj = base['image']['nobjects']
        if not isinstance(nobj, dict) or not nobj.get('_setup_as_list', False):
            logger.debug("generating nobj for all tiles:")
            seed = galsim.config.ParseValue(base['image'], 'random_seed', base, int)[0]
            base['tile_num_rng'] = base['rng'] = galsim.BaseDeviate(seed)
            nobj_list = []
            for tile_num in range(ntiles):
                base['tile_num'] = tile_num
                nobj = galsim.config.ParseValue(base['image'], 'nobjects', base, int)[0]
                nobj_list.append(nobj)
            base['image']['nobjects'] = {
                'type': 'List',
                'items': nobj_list,
                'index_key': 'tile_num',
                '_setup_as_list': True,
            }
        logger.debug('nobjects = %s', galsim.config.CleanConfig(base['image']['nobjects']))

        # base['image']['nobjects'] = None # TODO add here the


        # TODO We only need a noise seed, as the objects are already generated
        # TODO this should be greatly simplified and reduced
        # Set the random numbers to repeat for the objects so we get the same objects in the field
        # each time. In fact what we do is generate three sets of random seeds:
        # 0 : Sequence of seeds that iterates with obj_num i.e. no repetetion. Used for noise
        # 1 : Sequence of seeds that starts with the first object number for a given tile, then iterates
        # with the obj_num minus the first object number for that band, intended for quantities
        # that should be the same between bands for a given tile.
        rs = base['image']['random_seed']
        if not isinstance(rs, list):
            first = galsim.config.ParseValue(base['image'], 'random_seed', base, int)[0]
            base['image']['random_seed'] = []
            # The first one is the original random_seed specification, used for noise, since
            # that should be different for each band, and probably most things in input, output,
            # or image.
            if isinstance(rs, int):
                base['image']['random_seed'].append(
                    {'type': 'Sequence', 'index_key': 'obj_num', 'first': first})
            else:
                base['image']['random_seed'].append(rs)

            # The second one is used for the galaxies and repeats through the same set of seed
            # values for each band in a tile.
            if nobj > 0:
                base['image']['random_seed'].append(
                    {
                        'type': 'Eval',
                        'str': 'first + tile_start_obj_num + (obj_num - tile_start_obj_num) % nobjects',
                        'ifirst': first,
                        'inobjects': {'type': 'Current', 'key': 'image.nobjects'}
                    }
                )
            else:
                base['image']['random_seed'].append(base['image']['random_seed'][0])
            # The third iterates per tile
            base['image']['random_seed'].append(
                {'type': 'Sequence', 'index_key': 'tile_num', 'first': first})
            if 'gal' in base:
                base['gal']['rng_num'] = 1
            if 'stamp' in base:
                base['stamp']['rng_num'] = 1
            if 'image_pos' in base['image']:
                base['image']['image_pos']['rng_num'] = 1
            if 'world_pos' in base['image']:
                base['image']['world_pos']['rng_num'] = 1
        logger.debug('random_seed = %s', galsim.config.CleanConfig(base['image']['random_seed']))


        # TODO This is good
        bands = config["bands"]
        nbands = len(config["bands"])
        # Make sure that band_num is setup properly in the right places.
        tile_num = file_num // nbands
        band_num = file_num % nbands
        base['tile_num'] = tile_num
        base['band_num'] = band_num
        galsim.config.eval_base_variables += ["band"]
        base['band'] = config['bands'][band_num]
        print("tile_num", tile_num, "band_num", band_num)

        # TODO this should be removed, nobjects is no longer in "image"
        nobjects = galsim.config.ParseValue(base['image'], 'nobjects', base, int)[0]
        # print(base["image"]["nobjects"])
        # print(band_num, nobjects)

        # tile_start_obj_num is the object number of the first object in the current tile
        base['tile_start_obj_num'] = base['start_obj_num'] - band_num * nobjects

        # store file info in config
        # config["_file_names"] = {}
        # for tile_num in range(ntiles):
        #    config["_file_names"][tile_num] = {}
        #    for band_num in range(nbands):
        #        config["_file_names"][tile_num][band_num] = {}

        logger.debug('file_num, ntiles, nband = %d, %d, %d', file_num, ntiles, nbands)
        logger.debug('tile_num, band_num = %d, %d', tile_num, band_num)


        # TODO this is good
        # This stays as it is
        # This sets up the RNG seeds.
        OutputBuilder.setup(self, config, base, file_num, logger)

    def getNFiles(self, config, base):
        ntiles = galsim.config.ParseValue(config, 'ntiles', base, int)[0]
        nbands = len(config["bands"])
        config["nbands"] = nbands
        return ntiles * nbands

    def buildImages(self, config, base, file_num, image_num, obj_num, ignore, logger):
        """Build the images

        @param config           The configuration dict for the output field.
        @param base             The base configuration dict.
        @param file_num         The current file_num.
        @param image_num        The current image_num.
        @param obj_num          The current obj_num.
        @param ignore           A list of parameters that are allowed to be in config that we can
                                ignore here.  i.e. it won't be an error if they are present.
        @param logger           If given, a logger object to log progress.

        @returns a list of the images built
        """
        print("--------------------")
        print("buildImages IS CALLED")

        logger.info('Starting buildImages')
        logger.info('file_num: %d'%base['file_num'])
        logger.info('image_num: %d',base['image_num'])

        tile_num = base['tile_num']
        band_num = base['band_num']
        req = { 'nbands' : int, }
        opt = { 'ntiles' : int, }
        ignore += [ 'file_name', 'dir', 'bands' ]
        kwargs = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)[0]
        ntiles = kwargs.get('ntiles',1)
        nbands = kwargs['nbands']
        logger.error("ntiles, nbands, tile_num, band_num = %d, %d, %d, %d",ntiles,nbands,tile_num,band_num)

        #Save some stuff to the file_names dict
        if "_file_names" not in config:
            config["_file_names"] = {}
        if tile_num not in config["_file_names"]:
            config["_file_names"][tile_num] = {}
        if band_num not in config["_file_names"][tile_num]:
            config["_file_names"][tile_num][band_num] = {}
        if "tilename" not in config["_file_names"][tile_num]:
            config["_file_names"][tile_num]["tilename"] = galsim.config.GetCurrentValue( "eval_variables.stilename", base, str )

        file_names = config["_file_names"][tile_num][band_num]
        image_file = galsim.config.GetCurrentValue( "file_name", config, str, base )
        #Add directory if specified in config
        if "dir" in config:
            d = galsim.config.GetCurrentValue( "dir", config, str, base )
            image_file = os.path.join( d, image_file )
        #Make sure absolute path
        if not os.path.isabs(image_file):
            image_file = os.path.join( os.getcwd(), image_file )
        file_names["image_file"] = image_file

        if "truth" in config:
            truth_file = galsim.config.GetCurrentValue( "truth.file_name", config, str, base )
            if "dir" in config:
                truth_file = os.path.join( d, truth_file )
            #Make sure absolute path
            if not os.path.isabs(truth_file):
                truth_file = os.path.join( os.getcwd(), truth_file )
            file_names["truth"] = truth_file

        if base["psf"]["type"] == "DES_PSFEx":
            file_names["psfex_file"] = galsim.config.GetCurrentValue( "des_psfex.file_name", base["input"], str, base )
        if base["image"]["noise"]["type"] == "FitsNoise":
            file_names["weight"] = ( galsim.config.GetCurrentValue( "noise.file_name", base["image"], str, base ),
                                                                           galsim.config.GetCurrentValue( "noise.hdu", base["image"], int, base ) )
        elif "weight" in config:
            if "file_name" in config["weight"]:
                weight_file_name = ( galsim.config.GetCurrentValue( "weight.file_name", config, str, base ) )
                hdu = 0
            if "dir" in config["weight"]:
                d = ( galsim.config.GetCurrentValue( "weight.dir", config, str, base ) )
                weight_file_name = os.path.join(d, weight_file_name)
            else:
                weight_file_name = image_file
                hdu = ( galsim.config.GetCurrentValue( "weight.hdu", config, int, base ) )
            file_names["weight"] = ( weight_file_name, hdu )

        if "badpix" in config:
            if "file_name" in config["badpix"]:
                mask_file_name = ( galsim.config.GetCurrentValue( "badpix.file_name", config, str, base ) )
                hdu = 0
            if "dir" in config["badpix"]:
                d = ( galsim.config.GetCurrentValue( "badpix.dir", config, str, base ) )
                mask_file_name = os.path.join(d, mask_file_name)
            else:
                mask_file_name = image_file
                hdu = ( galsim.config.GetCurrentValue( "badpix.hdu", config, int, base ) )
            file_names["mask"] = ( mask_file_name, hdu )


        # Now we run the base class BuildImages, which just builds a single image.
        ignore += ['ntiles', 'nbands']
        images = OutputBuilder.buildImages(self, config, base, file_num, image_num, obj_num,
                                           ignore, logger)
        return images



galsim.config.output.RegisterOutputType('MultibandCatalogFits',  CatalogOutputBuilder())


import galsim.config.output