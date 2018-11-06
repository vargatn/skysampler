"""
Module to handle survey data Processing and IO
"""


import glob
import numpy as np
import pandas as pd
import fitsio as fio
import healpy as hp

from .utils import to_pandas, radial_bins
from .paths import setup_logger, logfile_info, config


BADVAL = -9999.
logger = setup_logger("INDEXER", level=config["logging_level"], logfile_info=logfile_info)


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


class TargetData(object):
    def __init__(self, fname, mode=None):
        """
        Simple wrapper for unified handling of clusters and random point tables

        Exposes richness, redshift, ra, dec columns

        Supports subselecting in richness and redshift

        Parameters
        ----------
        fname: str
            File name for fits table to use
        mode: str
            "clust" or "rands", if None figures out automatically
        """

        self.fname = fname
        self.mode = mode

        _data = fio.read(self.fname)
        self.alldata = to_pandas(_data)
        self.data = self.alldata

        self.inds = None
        self.pars = None
        self.limits = None
        self.assign_values()

    def assign_values(self):
        """Tries to Guess 'mode' and exposes richness and redshift columns"""
        if self.mode is not None:
            if self.mode == "clust":
                self.richness = self.data.LAMBDA_CHISQ
                self.redshift = self.data.Z_LAMBDA
            elif self.mode == "rands":
                self.richness = self.data.AVG_LAMBDAOUT
                self.redshift = self.data.ZTRUE
        else:
            try:
                self.richness = self.data.LAMBDA_CHISQ
                self.redshift = self.data.Z_LAMBDA
                self.mode = "clust"
            except:
                self.richness = self.data.AVG_LAMBDAOUT
                self.redshift = self.data.ZTRUE
                self.mode = "rands"

    def reset_data(self):
        """Resets data to original table"""
        self.data, self.inds = self.alldata, None
        self.assign_values()

    def draw_subset(self, nrows, rng=None):
        """draw random to subset of rows"""
        self.data, self.inds = subsample(self.data, nrows, rng=rng)
        self.assign_values()

    def select_inds(self, inds, bool=True):
        """"
        Selects subset based on index

        Parameters
        ----------
        inds: np.array
            indexing array
        bool: bool
            whether indexing array is bool ir integer
        """

        if bool:
            self.inds = np.nonzero(inds)
        else:
            self.inds = inds

        self.data = self.alldata.iloc[self.inds]

    def select_range(self, pars, limits):
        """
        Selects single parameter bin from underlying data table

        in addition to columns, "redshift" and "richness" are also valid keys

        Parameters
        ----------
        pars: str or list
            Column name or list of Column names
        limits: list
            value limits for each column

        """

        self.reset_data()

        self.pars = pars
        self.limits = limits

        bool_inds = np.ones(len(self.data), dtype=bool)
        for par, lim in zip(pars, limits):
            if par == "redshift":
                _ind = (self.redshift > lim[0]) & (self.redshift < lim[1])
                bool_inds[np.invert(_ind)] = False
            elif par == "richness":
                _ind = (self.richness > lim[0]) & (self.richness < lim[1])
                bool_inds[np.invert(_ind)] = False
            else:
                _ind = (self.alldata[par] > lim[0]) & (self.alldata[par] < lim[1])
                bool_inds[np.invert(_ind)] = False

        self.inds = np.nonzero(bool_inds)
        self.data = self.alldata.iloc[self.inds]
        self.assign_values()

    def to_dict(self):
        """Extracts metadata of self int a dictionary for lean storage"""
        info = {
            "fname": self.fname,
            "inds": self.inds,
            "pars": self.pars,
            "limits": self.limits,
            "mode": self.mode,
        }
        return info

    @classmethod
    def from_dict(cls, info):
        """recreate full object from dictionary"""
        res = cls(info["fname"], info["mode"])

        # reconstruct
        if (info["pars"] is None) & (info["inds"] is not None):
            res.select_inds(info["inds"], bool=False)
        if (info["pars"] is not None):
            res.select_range(info["pars"], info["limits"])

        return res

    @classmethod
    def from_config(cls, mode, config):
        """
        Automatically reads from config

        Parameters
        ----------
        mode: str
            clust or rands
        config: dict
            Config dictionary
        """

        if mode == "clust":
            fname = config["catalogs"]["targets"]["clust"]
        elif mode == "rands":
            fname = config["catalogs"]["targets"]["rands"]
        else:
            raise KeyError("Currently only clust and rands mode is supported")

        return cls(fname, mode)


class SurveyData(object):
    def __init__(self, fname_expr, nside):

        self.fname_expr = fname_expr
        self.nside = nside

        logger.critical("initated SurveyData")

    def convert_on_disk(self, suffix_in=".fits", suffix_out=".h5"):
        "convert all survey FITS files to PANDAS"

        fnames = np.sort(glob.glob(self.fname_expr + suffix_in))

        for fname in fnames:
            logger.critical("converting to pandas" + fname)
            data = fio.read(fname)
            data = to_pandas(data)
            data["IPIX"] = hp.ang2pix(self.nside, data.RA, data.DEC, lonlat=True)
            data.nside = self.nside
            data.to_hdf(fname.replace(suffix_in, suffix_out), key="data")

    def read_all_pandas(self, suffix=".h5"):
        """Reads all DataFrames to memory"""
        fnames = np.sort(glob.glob(self.fname_expr + suffix))

        datas = []
        for fname in fnames:
            logger.critical("loading " + fname)
            datas.append(pd.read_hdf(fname, key="data"))
        self.data = pd.concat(datas, ignore_index=True)
        self.nrows = len(self.data)

