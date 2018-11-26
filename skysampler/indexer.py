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


def get_theta_edges(nbins, theta_min, theta_max, eps):
    """
    Creates logarithmically space angular bins which include +- EPS linear range around zero

    The binning scheme looks the following::

        theta_edges = [ -eps, eps, theta_min, ... , theta_max]

    hence there are in total :code:`nbins + 2` bins.


    Parameters
    ----------
    nbins: int
        number of radial bins
    theta_min: float
        start of log10 spaced bins
    theta_max: float
        end of log10 spaced bins
    eps: float
        linear padding around zero

    Returns
    -------
    theta_edges: np.array
        radial edges
    rcens: np.array
        cemters of radial rings (starting at theta_min)
    redges: np.array
        edges of radial rings (starting at theta_min)
    rareas: np.array
        2D areas of radial rings (starting at theta_min)
    """
    rcens, redges, rareas = radial_bins(theta_min, theta_max, nbins)
    theta_edges = np.concatenate((np.array([-eps, eps, ]), redges))
    logger.debug("theta_edges " + str(theta_edges))
    logger.debug("rcens " + str(rcens))
    logger.debug("redges " + str(redges))
    logger.debug("rareas " + str(redges))
    return theta_edges, rcens, redges, rareas


def shuffle(tab, rng):
    """
    Returns a shuffled version of the passed DataFrame

    Uses :py:meth:`subsample` in the backend

    Parameters
    ----------
    tab: pd.DataFrame
        input table
    rng: np.random.RandomState
        random number generator, if None uses np.random directly

    Returns
    -------
    pd.DataFrame
        shuffled table
    """
    logger.debug("shuffling table in place")
    return subsample(tab, len(tab), rng, replace=False)


def get_ndraw(nsample, nchunk):
    """
    TODO
    """
    division = float(nsample) / float(nchunk)
    arr = np.array([int(round(division * (i+1))) - int(round(division * (i)))
                for i in range(nchunk) ])
    return arr


def subsample(tab, nrows=1000, rng=None, replace=False):
    """
    Choose rows randomly from pandas DataFrame

    Parameters
    ----------
    tab: pd.DataFrame
        input table
    nrows: int
        number of rows to choose, automatically capped at table length
    rng: np.random.RandomState
        random number generator, if None uses np.random directly
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
    logger.debug("subsampling " + str(nrows) + " objects out of " + str(len(tab)))
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
        logger.debug(self.fname)
        self.mode = mode
        logger.debug(self.mode)

        _data = fio.read(self.fname)
        self.alldata = to_pandas(_data)
        self.data = self.alldata
        logger.debug("data shape:" + str(self.alldata.shape))

        self.inds = None
        self.pars = None
        self.limits = None
        self.assign_values()
        logger.info("initiated TargetDate in mode " + str(self.mode) + " from " + str(self.fname))

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

        logger.debug("z: " + str(np.array(self.redshift)))
        logger.debug("lambda: " + str(np.array(self.richness)))
        self.ra = self.data.RA
        self.dec = self.data.DEC
        self.nrow = len(self.data)
        logger.info("Number of targets: " + str(self.nrow))

    def reset_data(self):
        """Resets data to original table"""
        self.data, self.inds = self.alldata, None
        self.assign_values()
        logger.info("resetting TargetData with filename " + str(self.fname))

    def draw_subset(self, nrows, rng=None):
        """draw random to subset of rows"""
        self.data, self.inds = subsample(self.data, nrows, rng=rng)
        self.assign_values()
        logger.info("drawing " + str(nrows) + " subset from  TargetData with filename " + str(self.fname))


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
        logger.info("selected inds (" + str(len(self.data)) + " subset) from  TargetData with filename " + str(self.fname))

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
        logger.info("selecting subset from  TargetData with filename " + str(self.fname))
        logger.info("pars:" + str(self.pars))
        logger.info("limits:" + str(self.limits))

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
        logger.info("constructing TargetData from config file")
        return cls(fname, mode)


class SurveyData(object):
    def __init__(self, fname_expr, nside):
        self.fname_expr = fname_expr
        self.nside = nside

        logger.info("initated SurveyData with expression " + str(self.fname_expr))

    def convert_on_disk(self, suffix_in=".fits", suffix_out=".h5"):
        "convert all survey FITS files to PANDAS"

        fnames = np.sort(glob.glob(self.fname_expr + suffix_in))
        logger.info("starting fits -> h5 conversion")
        for fname in fnames:
            logger.info("converting to pandas " + fname)
            data = fio.read(fname)
            data = to_pandas(data)
            data["IPIX"] = hp.ang2pix(self.nside, data.RA, data.DEC, lonlat=True)
            data.nside = self.nside
            data.to_hdf(fname.replace(suffix_in, suffix_out), key="data")
        logger.info("finished conversion")

    def read_all_pandas(self, suffix=".h5"):
        """Reads all DataFrames to memory"""
        fnames = np.sort(glob.glob(self.fname_expr + suffix))
        datas = []
        for fname in fnames:
            logger.info("loading " + fname)
            datas.append(pd.read_hdf(fname, key="data"))
        self.data = pd.concat(datas, ignore_index=True)
        self.nrows = len(self.data)
        logger.info("read " + str(self.nrows) + " rows")

    def drop_data(self):
        self.data = None
        self.nrows = None
        logger.info("resetting SurveyData with expression " + str(self.fname_expr))

    def lean_copy(self):
        return SurveyData(self.fname_expr, self.nside)


class SurveyIndexer(object):
    def __init__(self, survey, target, search_radius=360.,
                 nbins=50, theta_min=0.1, theta_max=100, eps=1e-3):
        self.survey = survey
        self.target = target

        self.search_radius = search_radius
        self.set_edges(nbins, theta_min, theta_max, eps)
        logger.info("Created SurveyIndexer")
        logger.debug(survey)
        logger.debug(target)
        logger.debug("search radius " + str(search_radius) + " arcmin")
        logger.info("nbins: " + str(nbins))
        logger.info("eps: " + str(eps) + " theta_min: " + str(theta_min) + " theta_max: " + str(theta_max))


    def set_edges(self, nbins=50, theta_min=0.1, theta_max=100, eps=1e-2):
        self.nbins = nbins
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.eps = eps
        self.theta_edges, self.rcens, self.redges, self.rareas = get_theta_edges(nbins, theta_min, theta_max, eps)

    def index(self):
        logger.info("starting survey indexing")
        self.numprof = np.zeros(self.nbins + 2)
        self.container = [[] for tmp in np.arange(self.nbins + 2)]
        for i in np.arange(self.target.nrow):
            logger.debug(str(i) + "/" + str(self.target.nrow))

            trow = self.target.data.iloc[i]
            tvec = hp.ang2vec(trow.RA, trow.DEC, lonlat=True)

            _radius = self.search_radius / 60. / 180. * np.pi
            dpixes = hp.query_disc(self.survey.nside, tvec, radius=_radius)

            gals = []
            for dpix in dpixes:
                cmd = "IPIX == " + str(dpix)
                gals.append(self.survey.data.query(cmd))
            gals = pd.concat(gals)

            darr = np.sqrt((trow.RA - gals.RA) ** 2. + (trow.DEC - gals.DEC) ** 2.) * 60. # converting to arcmin
            gals["DIST"] = darr

            tmp = np.histogram(darr, bins=self.theta_edges)[0]
            self.numprof += tmp

            for j in np.arange(self.nbins + 2):
                cmd = str(self.theta_edges[j]) + " < DIST < " + str(self.theta_edges[j + 1])
                rsub = gals.query(cmd)
                self.container[j].append(rsub.index.values)

        self.indexes, self.counts = [], []
        for j in np.arange(self.nbins):
            _uniqs, _counts = np.unique(np.concatenate(self.container[j]), return_counts=True)
            self.indexes.append(_uniqs)
            self.counts.append(_counts)

        result = IndexedDataContainer(self.survey.lean_copy(), self.target.to_dict(),
                                      self.numprof, self.indexes, self.counts,
                                      self.theta_edges, self.rcens, self.redges, self.rareas)
        logger.info("finished survey indexing")
        return result


    def draw_samples(self, nsample=10000, rng=None):
        logger.info("starting drawing random subsample with nsample=" + str(nsample))
        if rng is None:
            rng = np.random.RandomState()

        num_to_draw = np.min((self.numprof, np.ones(self.nbins+2)*nsample), axis=0).astype(int)
        limit_draw = num_to_draw == nsample

        self.sample_nrows = np.zeros(self.nbins + 2)
        samples = [[] for tmp in np.arange(self.nbins + 2)]
        self.samples = samples
        for i in np.arange(self.target.nrow):
            logger.debug(str(i) + "/" + str(self.target.nrow))

            trow = self.target.data.iloc[i]
            tvec = hp.ang2vec(trow.RA, trow.DEC, lonlat=True)

            _radius = self.search_radius / 60. / 180. * np.pi
            dpixes = hp.query_disc(self.survey.nside, tvec, radius=_radius)

            # pandas query
            gals = []
            for dpix in dpixes:
                cmd = "IPIX == " + str(dpix)
                gals.append(self.survey.data.query(cmd))
            gals = pd.concat(gals)

            darr = np.sqrt((trow.RA - gals.RA) ** 2. + (trow.DEC - gals.DEC) ** 2.) * 60. # converting to arcmin
            gals["DIST"] = darr

            digits = np.digitize(darr, self.theta_edges) - 1.
            gals["DIGIT"] = digits

            for d in np.arange(self.nbins+2):
                cmd = "DIGIT == " + str(d)
                rows = gals.query(cmd)

                # tries to draw a subsample from around each cluster in each radial range
                if limit_draw[d]:
                    _ndraw = get_ndraw(nsample - self.sample_nrows[d], self.target.nrow - i)[0]
                    ndraw = np.min((_ndraw, len(rows)))
                    self.sample_nrows[d] += ndraw
                    if ndraw > 0:
                        ii = rng.choice(np.arange(len(rows)), ndraw, replace=False)
                        samples[d].append(rows.iloc[ii])
                else:
                    # print("  ",len(rows))
                    self.sample_nrows[d] += len(rows)
                    samples[d].append(rows)

        for d in np.arange(self.nbins+2):
            self.samples[d] = pd.concat(samples[d], ignore_index=True)

        result = IndexedDataContainer(self.survey.lean_copy(), self.target.to_dict(),
                                      self.numprof, self.indexes, self.counts,
                                      self.theta_edges, self.rcens, self.redges, self.rareas,
                                      self.samples, self.sample_nrows)
        logger.info("finished random draws")
        return result


class IndexedDataContainer(object):
    def __init__(self, survey, target, numprof, indexes, counts, theta_edges, rcens, redges, rareas,
                 samples=None, samples_nrows=None):
        """
        Container for Indexed Survey Data

        It serves only as a data wrapper which can be pickled easily. The bulk of the survey data or target data
        should not be contained, and can be dropped and recovered when necessary.

        All parameters are class variables with the same name.

        Parameters
        ----------
        survey: :py:meth:`SurveyData` instance
            Container for the survey data
        target: :py:meth:`TargetData` instance
            Container for the target data
        numprof: np.array
            number profile of objects around the targets
        indexes: list
            index of unique galaxies at each radial bin around targets
        counts: list of list
            multiplicity of unique galaxies at each radial bin around targets
        theta_edges: np.array
                radial edges
        rcens: np.array
                centers of radial rings (starting at theta_min)
        redges: np.array
            edges of radial rings (starting at theta_min)
        rareas: np.array
            2D areas of radial rings (starting at theta_min)
        samples: list of pd.DataFrame
            table of random galaxy draws from each radial bin (capped in size at :code:`nsamples`)
        samples_nrows: np.array
            number of galaxies drawn from each radial bin
        """
        self.survey = survey
        self.target = target
        self.numprof = numprof
        self.indexes = indexes
        self.counts = counts
        self.theta_edges = theta_edges
        self.rcens = rcens
        self.redges = redges
        self.rareas = rareas
        self.samples = samples
        self.samples_nrows = samples_nrows

    def expand_data(self):
        """Recover all data from disk"""
        self.target = TargetData.from_dict(self.target)
        self.survey.read_all()

    def drop_data(self):
        """Drops all data and keeps only necessary values"""
        self.survey = self.survey.drop_data()
        self.target = self.target.to_dict()