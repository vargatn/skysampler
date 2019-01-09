# Copyright (c) 2012-2018 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import galsim
import logging

# This file adds image type Scattered, which places individual stamps at arbitrary
# locations on a larger image.

# TODO Rewrite this package to comply with skysampler while setting up the images.
# TODO the spirit should be that number of objects and current tile info is taken from the DATA FITS file

# TODO I think everything not related to this particular situation should be removed, as we won't be
# TODO able to catch all edge cases

# from galsim.config.image import ImageBuilder
from galsim.config.image_scattered import ScatteredImageBuilder
class ScatteredCatalogBuilder(ScatteredImageBuilder):

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        """
        # TODO rewrite documentation

        Do the initialization and setup for building the image.

        This figures out the size that the image will be, but doesn't actually build it yet.

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.
        @param obj_num      The first object number in the image.
        @param ignore       A list of parameters that are allowed to be in config that we can
                            ignore here. i.e. it won't be an error if these parameters are present.
        @param logger       If given, a logger object to log progress.

        @returns xsize, ysize
        """

        logger.debug('image %d: Building Scattered: image, obj = %d,%d',
                     image_num,image_num,obj_num)

        self.sampler = galsim.config.GetInputObj('sky_sampler', config, base, 'sky_sampler')

        self.tile_num = base["tile_num"]
        self.sampler.set_tile_num(self.tile_num)
        self.nobjects = self.getNObj(config, base, image_num)

        logger.debug('image %d: nobj = %d',image_num,self.nobjects)

        # These are allowed for Scattered, but we don't use them here.
        extra_ignore = [ 'image_pos', 'world_pos', 'stamp_size', 'stamp_xsize', 'stamp_ysize',
                         'nobjects' ]
        opt = { 'size' : int , 'xsize' : int , 'ysize' : int }
        params = galsim.config.GetAllParams(config, base, opt=opt, ignore=ignore+extra_ignore)[0]

        size = params.get('size',0)
        full_xsize = params.get('xsize',size)
        full_ysize = params.get('ysize',size)

        if (full_xsize <= 0) or (full_ysize <= 0):
            raise galsim.GalSimConfigError(
                "Both image.xsize and image.ysize need to be defined and > 0.")

        # If image_force_xsize and image_force_ysize were set in config, make sure it matches.
        if ( ('image_force_xsize' in base and full_xsize != base['image_force_xsize']) or
             ('image_force_ysize' in base and full_ysize != base['image_force_ysize']) ):
            raise galsim.GalSimConfigError(
                "Unable to reconcile required image xsize and ysize with provided "
                "xsize=%d, ysize=%d, "%(full_xsize,full_ysize))

        return full_xsize, full_ysize

    def getNObj(self, config, base, image_num):
        # FIXME reformat this documentation
        """Get the number of objects that will be built for this image.

        @param config       The configuration dict for the image field.
        @param base         The base configuration dict.
        @param image_num    The current image number.

        @returns the number of objects
        """
        orig_index_key = base.get('index_key',None)
        base['index_key'] = 'image_num'
        base['image_num'] = image_num

        nobj = self.sampler.get_nobj()
        base['index_key'] = orig_index_key
        return nobj

# Register this as a valid image type
from galsim.config.image import RegisterImageType
RegisterImageType('ScatteredCatalog', ScatteredCatalogBuilder())

