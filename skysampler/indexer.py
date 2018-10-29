"""
subsampler


index maker
loop throught clusters
define index and multiplitcity for each radial shell
container for indexed shells
query funciton for indexed shells to construct parameters space
feature space container (dynamic, and in memory)

The end goal is to have a wrapper class for indexing up surveys and processing them into feature spaces in a
object oriented way,

We should control it from a config file.


"""

import glob
import fitsio as fio
import numpy as np
import pandas as pd
import healpy as hp
import pickle
import astropy.units as u

from .utils import to_pandas
from .paths import setup_logger, logfile_info, config

logger = setup_logger("INDEXER", level=config["logging_level"], logfile_info=logfile_info)

BADVAL = -9999.

# TODO currently only implement in Memory version


def subsample(tab, nrows=1e3, rng=None, replace=False):
    """
    Choose rows randomly from pandas DataFrame

    Parameters
    ----------
    tab: pd.DataFrame
        input table
    nrows: int
        number of rows to choose, automatically capped at table length
    rng: np.random.RandomState
        random number generator
    replace: bool
        draw with replacement or not

    Returns
    -------
    pd.DataFrame
        random row subset of input table

    """

    if rng is None:
        rng = np.random.RandomState()

    nrows=np.min((len(tab), int(round(nrows))))
    allinds = np.arange(len(tab))
    inds = allinds[rng.choice(allinds, nrows, replace=replace)]
    return tab.iloc[inds]


class SurveyData(object):
    def __init__(self, fnames, nside):
        """
        Interface for Survey Data on disk
        """

        self.nside = nside
        self.fnames = fnames
        self.columns = fio.FITS(self.fnames[0])[1]
        self.data = None
        self.ra = None
        self.dec = None
        self.ipix = None

        logger.critical("initated SurveyData")

    @classmethod
    def from_config(cls, config):
        """Creates object based on config file"""
        all_fnames = np.sort(glob.glob(config["catalogs"]["survey"]["wide_data_expr"]))
        fnames = all_fnames[config["catalogs"]["survey"]["chunk_min"]:config["catalogs"]["survey"]["chunk_max"]]

        return cls(fnames=fnames, nside=config["catalogs"]["survey"]["nside"])

    def read_all(self):
        """Reads all data to memory"""
        data = []
        for fname in self.fnames:
            data.append(fio.read(fname))
        data = np.hstack(data)
        self.data = to_pandas(data)

        self.ra = self.data.RA
        self.dec = self.data.DEC

        self.data["IPIX"] = hp.ang2pix(self.nside, self.ra, self.dec, lonlat=True)
        self.ipix = self.data.IPIX
        logger.critical("Read Survey data to memory")


class TargetData(object):
    def __init__(self, fname):
        """
        Simple wrapper for unified handling of clusters and random point tables

        Exposes richness, redshift, ra, dec columns

        Parameters
        ----------
        fname: str
            File name for fits table to use

        """
        _data = fio.read(fname)
        self.data = to_pandas(_data)

        try:
            self.richness = self.data.LAMBDA_CHISQ
            self.redshift = self.data.Z_LAMBDA
            self.mode = "clust"
        except:
            self.richness = self.data.AVG_LAMBDAOUT
            self.redshift = self.data.ZTRUE
            self.mode = "rands"

        self.ra = self.data.RA
        self.dec = self.data.DEC

        logger.critical("initated TargetData in " + self.mode + " mode")

    @classmethod
    def from_config(cls, mode, config):
        """
        Automatically reads from config

        Parameters
        ----------
        mode: str
            clust or rands

        """

        if mode == "clust":
            fname = config["catalogs"]["targets"]["clust"]
        elif mode == "rands":
            fname = config["catalogs"]["targets"]["rands"]
        else:
            raise KeyError("Currently only clust and rands mode is supported")

        return cls(fname)


class SurveyIndexer(object):
    """
    Do the indexing of the Data based on passed cluster DataFrame


    This is mostly based on existing implementations

    What should be produced is the indexes of objects in radial shells, and their multiplicities.
    """
    def __init__(self):
        pass

    def index(self):

        pass

    # corrprof = np.zeros(len(redges) - 1)
    # radius = 6. / 180. * np.pi
    #
    # container = [[] for tmp in np.arange(len(redges) - 1)]
    # for i in np.arange(len(ctab)):
    #     print(i, "/", len(ctab), end="\r")
    #     clust = ctab.iloc[i]
    #
    #     # disc query pixes around cluster
    #     cpix = hp.ang2pix(nside, clust.RA, clust.DEC, lonlat=True)
    #     dpixes = hp.query_disc(nside, hp.pix2vec(nside, cpix), radius=radius, )
    #
    #     # pandas query
    #     gals = []
    #     for dpix in dpixes:
    #         cmd = "IPIX == " + str(dpix)
    #         gals.append(table.query(cmd))
    #     gals = pd.concat(gals)
    #
    #     try:
    #         zclust = clust.Z_LAMBDA
    #     except:
    #         zclust = clust.ZTRUE
    #
    #     # distances
    #     scale = 60.
    #     darr = np.sqrt((+ clust.RA - gals.RA) ** 2. + (clust.DEC - gals.DEC) ** 2.) * scale
    #     gals["DIST"] = darr
    #     corrprof += np.histogram(darr, bins=redges)[0]
    #
    #     for j in np.arange(len(redges) - 1):
    #         cmd = str(redges[j]) + " < DIST < " + str(redges[j + 1])
    #         rsub = gals.query(cmd)
    #         container[j].append(rsub.index.values)
    #
    # # reformatting results
    # indexes, counts = [], []
    # for j in np.arange(len(redges) - 1):
    #     _uniqs, _counts = np.unique(np.concatenate(container[j]), return_counts=True)
    #     indexes.append(_uniqs)
    #     counts.append(_counts)


class IndexedDataContainer(object):
    """Container for Indexed Data"""
    def __init__(self):
        pass

    def query(self):
        """
        Construct Parameter Space based on indexed data

        Ideally this also doubles as a shorthand we can use for visualizations
        """
        pass

    def write(self):
        pass

    def load(self):
        pass


class FeatureSpaceContainer(object):
    """
    Container for Feature space (should be very fine histogram, to reduce memory size

    We will use it for later processing, and for this reason should be completely self-standing, and not memory heavy
    """
    def __init__(self):
        pass

    def write(self):
        pass

    def load(self):
        pass


def extract_features():
    """
    Do the above using the classes

    Ideally should only take minimal arguements
    """
    pass






