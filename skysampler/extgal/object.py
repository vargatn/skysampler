import galsim
import ngmix
import numpy as np






def _mock_bdf(config, base, ignore, gsparams, logger):

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

galsim.config.RegisterObjectType('MockBDF', _mock_bdf)

