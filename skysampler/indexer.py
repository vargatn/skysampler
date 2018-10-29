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

from .utils import to_pandas, radial_bins
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
    return tab.iloc[inds], inds


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
        self.nrows = None

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
        self.nrows = len(self.data)

        self.ra = self.data.RA
        self.dec = self.data.DEC

        self.data["IPIX"] = hp.ang2pix(self.nside, self.ra, self.dec, lonlat=True)
        self.ipix = self.data.IPIX
        logger.critical("Read Survey data to memory")

    def drop_data(self):
        self.data = None
        self.ra = None
        self.dec = None
        self.ipix = None
        self.nrows = None

    def lean_copy(self):
        return SurveyData(self.fnames, self.nside)


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
        self.fname = fname
        _data = fio.read(self.fname)
        self.alldata = to_pandas(_data)
        self.data = self.alldata

        self.inds = None
        self.richness_min = None
        self.richness_max = None
        self.z_min = None
        self.z_max = None

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
        self.nrows = len(self.data)

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

    def assign_values(self):

        if self.mode == "clust":
            self.richness = self.data.LAMBDA_CHISQ
            self.redshift = self.data.Z_LAMBDA
        elif self.mode == "rands":
            self.richness = self.data.AVG_LAMBDAOUT
            self.redshift = self.data.ZTRUE
        else:
            raise KeyError("Currently only clust and rands mode is supported")

        self.ra = self.data.RA
        self.dec = self.data.DEC
        self.nrows = len(self.data)

    def select_subset(self, nrows, rng=None):
        """Restrict to subset of rows"""

        self.data, self.inds = subsample(self.data, nrows, rng=rng)
        self.assign_values()

    def select_range(self, config):

        self.richness_min = config["target_bins"]["richness_min"]
        self.richness_max = config["target_bins"]["richness_max"]
        self.z_min = config["target_bins"]["z_min"]
        self.z_max = config["target_bins"]["z_max"]

        selinds = ((self.richness > self.richness_min) &
                   (self.richness < self.richness_max) &
                   (self.redshift > self.z_min) &
                   (self.redshift < self.z_max))

        self.inds = np.nonzero(selinds)[0]
        self.data = self.data.iloc[self.inds]
        self.assign_values()

    def reset_data(self):
        self.data, self.inds = self.alldata, None
        self.assign_values()

    def to_dict(self):
        """Copy only filename and indexes"""

        res = {
            "fname": self.fname,
            "inds": self.inds,
            "richness_min": self.richness_min,
            "richness_max": self.richness_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }
        return res

    @classmethod
    def from_dict(cls, info):
        """recreate full object from lean copy"""
        res = TargetData(info["fname"])

        res.inds = info["inds"]
        res.redshift_min = info["redshift_min"]
        res.redshift_max = info["redshift_max"]
        res.z_min = info["z_min"]
        res.z_max = info["z_max"]

        res.data = res.alldata[res]
        res.assign_values()

        return res


class SurveyIndexer(object):
    def __init__(self, survey, target, config):
        """
        Do the indexing of the Data based on passed cluster DataFrame

        What should be produced is the indexes of objects in radial shells, and their multiplicities.
        """
        self.survey = survey
        self.target = target
        self.config = config

        self.theta_edges, self.rcens, self.redges, self.rareas = self.get_edges(config)
        self.nbins = len(self.theta_edges) - 1

        self.search_radius = self.config["indexer"]["search_radius"] * np.pi / 180.

    @staticmethod
    def theta_edges(eps, theta_min, theta_max, nbins):
        """Creates logarithmically space angular bins which include +- EPS linear range around zero"""
        rcens, redges, rareas = radial_bins(theta_min, theta_max, nbins)
        theta_edges = np.concatenate((np.array([-eps, eps, ]), redges))
        return theta_edges, rcens, redges, rareas

    def get_edges(self, config):
        """Construct linear-log edges from config file in ARCMIN"""
        eps = config["indexer"]["radial_bins"]["eps"]
        theta_min = config["indexer"]["radial_bins"]["theta_min"]
        theta_max = config["indexer"]["radial_bins"]["theta_max"]
        nbins = config["indexer"]["radial_bins"]["nbins"]
        return self.theta_edges(eps, theta_min, theta_max,  nbins)

    def index(self):
        """Calculates indexes and multiplicities for galaxies in radial shells"""
        numprof = np.zeros(self.nbins)
        container = [[] for tmp in np.arange(self.nbins)]
        for i in np.arange(self.target.nrow):
            # TODO replace this with logging
            print(i, "/", len(self.target.nrow), end="\r")
            trow = self.target.iloc[i]
            tvec = hp.ang2vec(trow.RA, trow.DEC, lonlat=True)

            dpixes = hp.query_disc(self.survey.nside, tvec, radius=self.search_radius)

            # pandas query
            gals = []
            for dpix in dpixes:
                cmd = "IPIX == " + str(dpix)
                gals.append(self.survey.data.query(cmd))
            gals = pd.concat(gals)

            darr = np.sqrt((trow.RA - gals.ra) ** 2. + (trow.DEC - gals.dec) ** 2.) * 60. # converting to arcmin
            gals["DIST"] = darr
            numprof += np.histogram(darr, bins=self.theta_edges)[0]

            for j in np.arange(self.nbins):
                cmd = str(self.theta_edges[j]) + " < DIST < " + str(self.theta_edges[j + 1])
                rsub = gals.query(cmd)
                container[j].append(rsub.index.values)

        # reformatting results
        indexes, counts = [], []
        for j in np.arange(self.nbins):
            _uniqs, _counts = np.unique(np.concatenate(container[j]), return_counts=True)
            indexes.append(_uniqs)
            counts.append(_counts)

        result = IndexedDataContainer(self.survey.lean_copy(), self.target.to_dict(), numprof, indexes, counts,
                                      self.theta_edges, self.rcens, self.redges, self.rareas)

        return result


class IndexedDataContainer(object):
    """Container for Indexed Data"""
    def __init__(self, survey, target_dict, numprof, indexes, counts, theta_edges, rcens, redges, rareas):
        self.survey = survey
        self.target_dict = target_dict
        self.numprof = numprof
        self.indexes = indexes
        self.counts = counts
        self.theta_edges = theta_edges
        self.rcens = rcens
        self.redges = redges
        self.rareas = rareas

    def query(self):
        """
        Construct Parameter Space based on indexed data

        Ideally this also doubles as a shorthand we can use for visualizations
        """
        pass





