"""
Should containe the Gaussian Processes operations


In addition to the feature spaces we should also take into account the average numbers of objects,
e.g. radial number profile (in absolute terms)


"""

import fitsio as fio
import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors
import sklearn.model_selection as modsel
import sklearn.preprocessing as preproc
import copy
import sys
import kdos #kde optimizer


ENDIANS = {
    "little": "<",
    "big": ">",
}

import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
except:
    mpl.use("Agg")
    import matplotlib.pyplot as plt

import multiprocessing as mp

from .utils import partition


def get_angle(num, rng):
    angle = rng.uniform(0, np.pi, size=num)
    return angle


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=0, weights=weights)
    variance = np.average((values-average)**2, axis=0, weights=weights)
    return np.sqrt(variance)


class BaseContainer(object):

    def __init__(self):
        self.alldata = None
        self.features = None
        self.weights = None

    def construct_features(self, columns, limits=None, logs=None):
        self.columns = columns
        self.limits = limits
        self.logs = logs
        self.features = pd.DataFrame()

        self.inds = np.ones(len(self.alldata), dtype=bool)
        for i, col in enumerate(columns):
            if isinstance(col[1], str):
                res = self.alldata[col[1]]
            else:
                if len(col[1]) == 3:
                    if isinstance(col[1][0], str):
                        col1 = self.alldata[col[1][0]]
                    elif isinstance(col[1][0], (list, tuple)):
                        col1 = self.alldata[col[1][0][0]][:, col[1][0][1]]
                    else:
                        col1 = col[1][0]

                    if isinstance(col[1][1], str):
                        col2 = self.alldata[col[1][1]]
                    elif isinstance(col[1][1], (list, tuple)):
                        col2 = self.alldata[col[1][1][0]][:, col[1][1][1]]
                    else:
                        col2 = col[1][1]

                    if col[1][2] == "-":
                        res = col1 - col2
                    elif col[1][2] == "+":
                        res = col1 + col2
                    elif col[1][2] == "*":
                        res = col1 * col2
                    elif col[1][2] == "/":
                        res = col1 / col2
                    elif col[1][2] == "SQSUM":
                        res = np.sqrt(col1**2. + col2**2.)
                    else:
                        raise KeyError("only + - * / are supported at the moment")

                elif len(col[1]) == 2:
                    res = self.alldata[col[1][0]][:, col[1][1]]
                else:
                    raise KeyError

            self.features[col[0]] = res.astype("float64")
            #
            if limits is not None:
                self.inds &= (self.features[col[0]] > limits[i][0]) & (self.features[col[0]] < limits[i][1])

        self.features = self.features[self.inds]

        try:
            self.weights = self.alldata["WEIGHT"][self.inds]
        except:
            self.weights = pd.Series(data=np.ones(len(self.features)), name="WEIGHT")

        for i, col in enumerate(columns):
            if logs is not None and logs[i]:
              self.features[col[0]] = np.log10(self.features[col[0]])

    def to_dual(self, **kwargs):
        res = DualContainer(self.features.columns, **kwargs)
        res.set_data(self.features, self.weights)
        return res


class FeatureSpaceContainer(BaseContainer):
    def __init__(self, info):
        """
        This needs to be done first
        """
        BaseContainer.__init__(self)

        self.rcens = info.rcens
        self.redges = info.redges
        self.rareas = info.rareas

        self.survey = info.survey
        self.target = info.target

        self.numprof = info.numprof
        self.samples = info.samples

        self.alldata = pd.concat(self.samples).reset_index(drop=True)

        self.nobj = self.target.nrow

    def surfdens(self, icol=0, scaler=1):
        if self.logs[icol]:
            arr = 10**self.features.values[:, icol]
        else:
            arr = self.features.values[:, icol]
        vals = np.histogram(arr, bins=self.redges, weights=self.weights)[0] / self.nobj / self.rareas * scaler
        return vals

    def downsample(self, nmax=10000, r_key="LOGR", nbins=40, rng=None, **kwargs):
        """Radially balanced downsampling"""

        if rng is None:
            rng = np.random.RandomState()

        rarr = self.features[r_key]
        # rbins = np.sort(rng.uniform(low=rarr.min(), high=rarr.max(), size=nbins+1))
        rbins = np.linspace(rarr.min(), rarr.max(), nbins+1)

        tmp_features = []
        tmp_weights = []
        for i, tmp in enumerate(rbins[:-1]):
            selinds = (self.features[r_key] > rbins[i]) & (self.features[r_key] < rbins[i + 1])
            vals = self.features.loc[selinds]
            ww = self.weights.loc[selinds]

            if len(vals) < nmax:
                tmp_features.append(vals)
                tmp_weights.append(ww)
            else:
                inds = np.arange(len(vals))
                pp = ww / ww.sum()
                chindex = rng.choice(inds, size=nmax, replace=False, p=pp)

                newvals = vals.iloc[chindex]
                newww = ww.iloc[chindex] * len(ww) / nmax

                tmp_features.append(newvals)
                tmp_weights.append(newww)

        features = pd.concat(tmp_features)
        weights = pd.concat(tmp_weights)

        res = DualContainer(features.columns, **kwargs)
        res.set_data(features, weights=weights)
        return res


class DeepFeatureContainer(BaseContainer):
    def __init__(self, data):
        BaseContainer.__init__(self)
        self.alldata = data
        self.weights = pd.Series(data=np.ones(len(self.alldata)), name="WEIGHT")

    @classmethod
    def from_file(cls, fname, flagsel=True):

        if ".fit" in fname:
            _deep = fio.read(fname)
        else:
            _deep = pd.read_hdf(fname, key="data").to_records()

        if flagsel:
            inds = _deep["flags"] == 0
            deep = _deep[inds]
        else:
            deep = _deep
        return cls(deep)


class DualContainer(object):
    """Contains features in normal and in transformed space"""
    def __init__(self, columns=None, mean=None, sigma=None, r_normalize=False, qt_params=None, r_key="LOGR", kde_transform=False, kde_trans_params=None):
        """
        One column Dataframes can be created by tab[["col"]]
        Parameters
        ----------
        columns
        mean
        sigma
        """
        self.columns = columns
        self.mean = mean
        self.sigma = sigma

        self.r_normalize = r_normalize
        self.r_key = r_key
        self.qt_params = qt_params
        self.qt = None

        self.kde_transform = kde_transform
        self.kde_transformer = kdos.kde_transformer(kde_trans_params=kde_trans_params)


    def __getitem__(self, key):
        if self.mode == "xarr":
            return self.xarr[key]
        else:
            return self.data[key]

    def set_mode(self, mode):
        self.mode = mode

    def set_xarr(self, xarr):
        self.xarr = pd.DataFrame(columns=self.columns, data=xarr).reset_index(drop=True)

        if self.r_normalize:
            self.qt = preproc.QuantileTransformer(output_distribution="normal")
            self.qt.set_params(**self.qt_params["params"])
            self.qt.quantiles_ = self.qt_params["quantiles"]
            self.qt.references_ = self.qt_params["references"]

            self.xarr[self.r_key] = self.qt.inverse_transform(self.xarr[self.r_key].values.reshape(-1, 1))

        self.data = self.xarr * self.sigma + self.mean
        self.weights = pd.Series(np.ones(len(self.data)), name="WEIGHT")
        self.shape = self.data.shape

    def set_data(self, data, weights=None):
        self.columns = data.columns
        self.data = data.reset_index(drop=True)
        self.weights = weights

        if self.weights is None:
            self.weights = pd.Series(np.ones(len(self.data)), name="WEIGHT")
            # self.weights["WEIGHT"] =

        # print(self.weights.shape)
        # print(self.data.shape)
        self.mean = np.average(self.data, axis=0, weights=self.weights)
        self.sigma = weighted_std(self.data, weights=self.weights)

        self.xarr = ((self.data - self.mean) / self.sigma)

        if self.r_normalize:
            self.qt = preproc.QuantileTransformer(output_distribution="normal")
            self.qt.fit(self.xarr[self.r_key].values.reshape(-1, 1))
            self.qt_params = {
                "params": self.qt.get_params(),
                "quantiles": self.qt.quantiles_,
                "references": self.qt.references_,
            }
            self.xarr[self.r_key] = self.qt.transform(self.xarr[self.r_key].values.reshape(-1, 1))

        if self.kde_transform == True:
            self.kde_transformer.initialize_transformer(xarr)
            self.kde_transformer.optimize_tranformer(n_steps=10)

        self.shape = self.data.shape
        self.mode = "data"

    def transform(self, arr):
        """From Data to Xarr"""

        if not isinstance(arr, pd.DataFrame):
            tab = pd.DataFrame(data=arr, columns=self.columns)
        else:
            tab = arr

        res = (tab - self.mean)/ self.sigma
        if self.r_normalize:
            res[self.r_key] = self.qt.transform(res[self.r_key].values.reshape(-1, 1))

        if self.kde_transform == True:
            res = self.kde_transformer.transform(res)
        return res

    def inverse_transform(self, arr):
        """From Xarr to Data"""

        if not isinstance(arr, pd.DataFrame):
            tab = pd.DataFrame(data=arr, columns=self.columns)
        else:
            tab = arr

        if self.kde_transform == True:
            tab = self.kde_transformer.inverse_transform(res)

        if self.r_normalize:
            tab[self.r_key] = self.qt.inverse_transform(tab[self.r_key].values.reshape(-1, 1))
        res = tab * self.sigma + self.mean
        return res

    def shuffle(self):
        self.sample(n=None, frac=1.)

    def sample(self, n=None, frac=1.):
        inds = np.arange(len(self.data))

        tab = pd.DataFrame()
        tab["IND"] = inds
        inds = tab.sample(n=n, frac=frac)["IND"].values

        self.data = self.data.iloc[inds].copy().reset_index(drop=True)
        self.weights = self.weights.iloc[inds].copy().reset_index(drop=True)
        self.xarr = self.xarr.iloc[inds].copy().reset_index(drop=True)
        self.shape = self.data.shape

    def set_rmax(self, rmax= 100., rkey="LOGR"):
        inds = self.data[rkey] < rmax

        res = DualContainer(**self.get_meta())
        res.data = self.data.loc[inds].copy().reset_index(drop=True)
        res.weights = self.weights.loc[inds].copy().reset_index(drop=True)
        res.xarr = self.xarr.loc[inds].copy().reset_index(drop=True)
        res.shape = res.data.shape
        return res

    def get_meta(self):
        info = {
            "columns": self.columns,
            "mean": self.mean,
            "sigma": self.sigma,
            "qt_params": self.qt_params,
            "r_key": self.r_key,
            "r_normalize": self.r_normalize
        }
        return info

    def match_surfdens(self):
        pass


def _add(a, b):
    return a + b

def _subtr(a, b):
    return a - b

_OPERATORS = {
    "+": _add,
    "-": _subtr,
}

class MultiEyeballer(object):
    """
    this needs to be done second
    """
    _radial_splits = np.logspace(np.log10(0.1), np.log10(40), 10)
    _cm_diagrams = (("MAG_R", "COLOR_G_R"), ("MAG_I", "COLOR_R_I"), ("MAG_Z", "COLOR_I_Z"))
    _cc_diagrams = (("COLOR_G_R", "COLOR_R_I"), ("COLOR_R_I", "COLOR_I_Z"))
    _plot_series = ("CC", "CM", "RR")
    _reconstr_mags = {
        "MAG_G": ("COLOR_G_R + COLOR_R_I + MAG_I"),
        "MAG_R": ("COLOR_R_I + MAG_I"),
        "MAG_Z": ("MAG_I - COLOR_I_Z"),
    }

    def __init__(self, container, radial_splits=None, cmap=None):
        """
        Constructs a large set of comparison images

        Density comparisons  and such

        """
        self.container = container
        self.cmap = cmap

        if radial_splits is not None:
            self._radial_splits = radial_splits

    def _get_col(self, label):
        if label in self.container.columns:
            col = self.container[label]
        elif label in self._reconstr_mags:
            tokens = self._reconstr_mags[label].split()
            col = self.container[tokens[0]]
            for i in np.arange(len(tokens) // 2):
                col = _OPERATORS[tokens[i + 1]](col, self.container[tokens[i + 2]])
        else:
            raise KeyError

        return col

    def radial_series(self, label1="MAG_I", label2="COLOR_R_I", rlabel="LOGR",
                      rlog=True, bins=None, fname=None, vmin=1e-3, vmax=None, nbins=60, title=None):

        if rlog:
            rr = 10 ** self.container[rlabel]
        else:
            rr = self.container[rlabel]

        col1 = self._get_col(label1)
        col2 = self._get_col(label2)

        ww = self.container.weights

        if bins is None:
            bins = (np.linspace(col1.min(), col1.max(), nbins),
                    np.linspace(col2.min(), col2.max(), nbins))

        num = len(self._radial_splits) - 1
        nrows = int(np.ceil(np.sqrt(num)))
        ncols = int(np.round(np.sqrt(num)))
        xsize = 4 * nrows
        ysize = 3 * ncols

        fig, axarr = plt.subplots(nrows=ncols, ncols=nrows, figsize=(xsize, ysize))
        faxarr = axarr.flatten()

        if title is not None:
            fig.text(0.125, 0.9, title, fontsize=14)

        for axs in axarr:
            axs[0].set_ylabel(label2)

        for axs in axarr[-1]:
            axs.set_xlabel(label1)

        for i, ax in enumerate(faxarr):
            ind = (rr > self._radial_splits[i]) & (rr < self._radial_splits[i + 1])
            if i == 0 and vmax is None:
                tmp = np.histogram2d(col1[ind], col2[ind], weights=ww[ind], bins=bins, normed=True)[0]
                vmax = 1 * tmp.max()
            ax.hist2d(col1[ind], col2[ind], weights=ww[ind], bins=bins, cmap=self.cmap,
                      norm=mpl.colors.LogNorm(), normed=True, vmax=vmax, vmin=vmin)
            ax.grid(ls=":")

            ax.text(0.05, 0.87,
                    "$R\in[{:.2f};{:.2f})$".format(self._radial_splits[i], self._radial_splits[i + 1]),
                    transform=ax.transAxes)

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches="tight")
        return fig, axarr

    def set_info(self, radial_splits=None, cm_diagrams=None, cc_diagrams=None, plot_series=None,
                 reconstr_mags=None):

        if radial_splits is not None:
            self._radial_splits = radial_splits
        if cc_diagrams is not None:
            self._cc_diagrams = cc_diagrams
        if cm_diagrams is not None:
            self._cm_diagrams = cm_diagrams
        if plot_series is not None:
            self._plot_series = plot_series
        if reconstr_mags is not None:
            self._reconstr_mags = reconstr_mags

    def corner(self, rbin=None, rlabel="LOGR", rlog=True, clognorm=True, bins=None, nbins=60,
               fname=None, vmin=None, vmax=None, title=None):
        # This should be one radial bin, or if None, then all



        if rbin is not None:
            if rlog:
                rr = 10 ** self.container[rlabel]
            else:
                rr = self.container[rlabel]
            rind = (rr > self._radial_splits[rbin]) & (rr < self._radial_splits[rbin + 1])
        else:
            rind = np.ones(len(self.container.data), dtype=bool)

        columns = list(self.container.columns)
        if bins is None:
            bins = []
            for col in columns:
                bins.append(np.linspace(self.container[col].min(),
                                        self.container[col].max(), nbins))

        nrows = len(columns)
        ncols = len(columns)
        xsize = 2.5 * nrows
        ysize = 2.5 * ncols

        fig, axarr = plt.subplots(nrows=nrows, ncols=nrows, figsize=(xsize, ysize))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        faxarr = axarr.flatten()

        if title is not None:
            fig.text(0.125, 0.9, title, fontsize=14)

        # blocking upper triangle
        for i in np.arange(nrows):
            for j in np.arange(ncols):
                if j < i:
                    norm = mpl.colors.Normalize()
                    if clognorm:
                        norm = mpl.colors.LogNorm()

                    axarr[i, j].hist2d(self.container[columns[j]][rind],
                                       self.container[columns[i]][rind],
                                       weights=self.container.weights[rind],
                                       bins=(bins[j], bins[i]), cmap=self.cmap,
                                       norm=norm, normed=True, vmax=vmax, vmin=vmin)
                    axarr[i, j].grid(ls=":")
                if i == j:
                    axarr[i, j].hist(self.container[columns[i]][rind], bins=bins[i],
                                     weights=self.container.weights[rind],
                                     histtype="step", density=True)
                    axarr[i, j].grid(ls=":")
                if j > i:
                    axarr[i, j].axis("off")
                if i < nrows - 1:
                    axarr[i, j].set_xticklabels([])
                if j > 0:
                    axarr[i, j].set_yticklabels([])
                if j == 0 and i > 0:
                    axarr[i, j].set_ylabel(columns[i])
                if i == nrows - 1:
                    axarr[i, j].set_xlabel(columns[j])

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches="tight")
        return fig, axarr

    def diffplot(self, fsc1, fsc2):
        # TODO this needs implemented in the end...

        #         this should be almost like the the other but some tweaking with the color scales
        pass

    def get_corner_bins(self, nbins=60):
        # corner bins
        self.corner_bins = []
        columns = list(self.container.columns)
        for col in columns:
            self.corner_bins.append(np.linspace(self.container[col].min(),
                                                self.container[col].max(), nbins))
        return self.corner_bins

    def plot_radial_ensemble(self, fname_root, cm=True, cc=True, radial=True, nbins=60, vmax=None, vmin=None,
                             cm_bins=None, cc_bins=None, corner_bins=None, title=None, suffix=""):
        """loop through a set of predefined diagrams"""

        # c-m diagrams
        if cm:
            for cm in self._cm_diagrams:
                fname = fname_root + "_cm_" + cm[0] + "_" + cm[1] + suffix + ".png"
                print(fname)
                self.radial_series(label1=cm[0], label2=cm[1], fname=fname,
                                   nbins=nbins, vmax=vmax, bins=cm_bins, title=title)

        # c-c diagrams
        if cc:
            for cc in self._cc_diagrams:
                fname = fname_root + "_cc_" + cc[0] + "_" + cc[1] + suffix + ".png"
                print(fname)
                self.radial_series(label1=cc[0], label2=cc[1], fname=fname,
                                   nbins=nbins, vmax=vmax, bins=cc_bins, title=title)

        # all -radial bin
        if radial:
            fname = fname_root + "_corner_all" + suffix + ".png"
            print(fname)
            self.corner(fname=fname, nbins=nbins, vmax=vmax, bins=corner_bins, title=title, vmin=vmin)

            # radial bins
            for rbin in np.arange(len(self._radial_splits) - 1):
                fname = fname_root + "_corner_rbin{:02d}".format(rbin) + suffix + ".png"
                print(fname)
                self.corner(rbin=rbin, fname=fname, nbins=nbins, vmax=vmax, bins=corner_bins, title=title, vmin=vmin)


# This is just a standard function
class FeatureEmulator(object):
    def __init__(self, container, rng=None):
        """This is just the packaged version of the KDE"""
        self.container = container
        self.kde = None

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng
    def train(self, bandwidth=0.2, kernel="tophat", atol=1e-6, rtol=1e-6, breadth_first=False, **kwargs):
            """train the emulator"""
            self.bandwidth = bandwidth
            self.kwargs = kwargs
            self.kde = neighbors.KernelDensity(bandwidth=self.bandwidth, kernel=kernel,
                                               atol=atol, rtol=rtol, breadth_first=breadth_first, **kwargs)
            self.kde.fit(self.container.xarr, sample_weight=self.container.weights)

    def draw(self, num, rmin=None, rmax=None, rcol="LOGR", mode="data"):
        """draws random samples from KDE maximum radius"""
        res = DualContainer(**self.container.get_meta())
        _res = self.kde.sample(n_samples=int(num), random_state=self.rng)
        res.set_xarr(_res)

        if (rmin is not None) or (rmax is not None):
            if rmin is None:
                rmin = res.data[rcol].min()

            if rmax is None:
                rmax = res.data[rcol].max()

            # these are the indexes to replace, not the ones to keep...
            inds = (res.data[rcol] > rmax) | (res.data[rcol] < rmin)
            while inds.sum():
            # print(inds.sum())
                vals = self.kde.sample(n_samples=int(inds.sum()), random_state=self.rng)
                _res[inds, :] = vals
                res.set_xarr(_res)
                inds = (res.data[rcol] > rmax) | (res.data[rcol] < rmin)

        res.set_mode(mode)
        return res

    def score_samples(self, arr):
        """Assuming that arr is in the data format"""

        arr = self.container.transform(arr)
        res = self.kde.score_samples(arr)
        return res

    def to_dict(self):


        # Memory view of training data cannot be pickled and is not strictly necessary
        _state = list(self.kde.tree_.__getstate__())
        sample_weights = copy.deepcopy(np.array(_state[-1]))


        # we have to change the state so that it can be pickled, it will be reconustructed later
        state = copy.deepcopy(_state[:-1])
        state.append(None)

        self.kde.tree_.__setstate__(state)
        res = {
            "bandwidth": self.bandwidth,
            "container_meta": self.container.get_meta(),
            "kde": copy.deepcopy(self.kde),
            "sample_weights": sample_weights,
        }
        # we have to reset the original state
        self.kde.tree_.__setstate__(_state)
        # print(res["sample_weights"])
        # print(np.array(self.kde.tree_.__getstate__()[-1]))
        return res

    @classmethod
    def from_dict(cls, info):
        container = DualContainer(**info["container_meta"])
        res = cls(container)
        res.kde = info["kde"]
        _state = list(res.kde.tree_.__getstate__())
        _state[-1] = info["sample_weights"]
        # _state[-2] = neighbors.dist_metrics.EuclideanDistance()
        res.kde.tree_.__setstate__(_state)
        return res

##########################################################################

def construct_wide_container(dataloader, settings, nbins=100, nmax=5000, r_normalize=False):
    fsc = FeatureSpaceContainer(dataloader)
    fsc.construct_features(**settings)
    # cont = fsc.to_dual(r_normalize=r_normalize)
    cont_small = fsc.downsample(nbins=nbins, nmax=nmax, r_normalize=r_normalize)
    cont_small.shuffle()
    return cont_small


def construct_deep_container(fname, settings, frac=1.):
    fsc = DeepFeatureContainer.from_file(fname)
    fsc.construct_features(**settings)
    cont = fsc.to_dual()
    cont.sample(frac=frac)
    return cont


##########################################################################

def make_infodicts(wcr, wr, dc, dsmc, nsamples, cols, nchunks=30, bandwidth=0.2, sample_rmax=5,
                   atol=1e-4, rtol=1e-4):
    wr_emu = FeatureEmulator(wr)
    wr_emu.train(kernel="tophat", bandwidth=bandwidth, atol=atol, rtol=rtol)

    dsmc_emu = FeatureEmulator(dsmc)
    dsmc_emu.train(kernel="tophat", bandwidth=bandwidth, atol=atol, rtol=rtol)

    _sample = dsmc_emu.draw(num=nsamples)

    rvals = wr_emu.draw(num=nsamples, rmax=sample_rmax)
    sample = pd.merge(_sample.data, rvals.data, left_index=True, right_index=True)
    sample_inds = partition(list(sample.index), nchunks)

    infodicts = []
    for i in np.arange(nchunks):
        subsample = sample.loc[sample_inds[i]]

        info = {
            "sample": subsample,
            "dc_cont": dc,
            "wr_cont": wr,
            "wcr_cont": wcr,
            "cols": cols,
            "bandwidth": bandwidth,
            "atol": atol,
            "rtol": rtol,
        }
        infodicts.append(info)
    return sample, infodicts


def run_scores(infodicts):
    pool = mp.Pool(processes=len(infodicts))
    try:
        pp = pool.map_async(calc_scores, infodicts)
        # the results here should be a list of score values
        result = pp.get(86400)  # apparently this counters a bug in the exception passing in python.subprocess...
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating workers")
        pool.terminate()
        pool.join()
    else:
        pool.close()
        pool.join()

    dc_scores = []
    wr_scores = []
    wcr_scores = []
    for res in result:
        dc_scores.append(res[0])
        wr_scores.append(res[1])
        wcr_scores.append(res[2])

    dc_scores = np.concatenate(dc_scores)
    wr_scores = np.concatenate(wr_scores)
    wcr_scores = np.concatenate(wcr_scores)

    return dc_scores, wr_scores, wcr_scores


def calc_scores(info):
    dc_score, wr_score, wcr_score = [], [], []
    try:
        sample = info["sample"]

        dc_emu = FeatureEmulator(info["dc_cont"])
        dc_emu.train(kernel="tophat", bandwidth=info["bandwidth"], atol=info["atol"], rtol=info["rtol"])

        wr_emu = FeatureEmulator(info["wr_cont"])
        wr_emu.train(kernel="tophat", bandwidth=info["bandwidth"], atol=info["atol"], rtol=info["rtol"])

        wcr_emu = FeatureEmulator(info["wcr_cont"])
        wcr_emu.train(kernel="tophat", bandwidth=info["bandwidth"], atol=info["atol"], rtol=info["rtol"])

        dc_score = dc_emu.score_samples(sample[info["cols"]["cols_dc"]])
        wr_score = wr_emu.score_samples(sample[info["cols"]["cols_wr"]])
        wcr_score = wcr_emu.score_samples(sample[info["cols"]["cols_wcr"]])

    except KeyboardInterrupt:
        pass

    return dc_score, wr_score, wcr_score


def to_buffer(arr):
    mp_arr = mp.Array("d", arr.flatten(), lock=False)
    return mp_arr, arr.shape


def from_buffer(mp_arr, shape):
    arr = np.frombuffer(mp_arr, dtype="d").reshape(shape)
    return arr


class KFoldValidator(object):
    """
    TODO validation package,

    Should automate splitting base data into training and test
    """
    def __init__(self, container, cv=5, param_grid=None, extra_params=None, score_train=False):
        self.container = container
        self.cv = cv

        self.param_grid = param_grid
        self.param_list = list(modsel.ParameterGrid(param_grid))
        if extra_params is None:
            self.extra_params = {}
        else:
            self.extra_params = extra_params

        # self.mp_xarr, self.mp_xarr_shape = to_buffer(self.container.xarr.values)
        # self.mp_w, self.mp_w_shape = to_buffer(self.container.weights.values)

        self._calc_split()

        self.result = None
        self.scores = None
        self.score_train = score_train

    def _calc_split(self):
        # TODO replace this with something that balances weights...

        inds = np.arange(len(self.container.data))
        # ww = self.container.weights

        kfold = modsel.KFold(n_splits=self.cv)
        self.splits = list(kfold.split(inds))

    def _get_infodicts(self):
        """
        Splits the dataset into a set of dictionaries dispathable to subprocesses
        """

        infodicts = []
        for i, params in enumerate(self.param_list):
            for j, split in enumerate(self.splits):
                info = {
                    "id": (i,j),
                    "split": split,
                    "params": params,
                    "extra_params": self.extra_params,
                    "xarr": self.container.xarr,
                    "w": self.container.weights,
                    "score_train": self.score_train,
                    # "meta": self.container.get_meta(),
                    # "mp_xarr": self.mp_xarr,
                    # "mp_xarr_shape": self.mp_xarr_shape,
                    # "mp_w": self.mp_w,
                    # "mp_w_shape": self.mp_w_shape,
                }
                infodicts.append(info)

        return infodicts

    def run(self, nprocess=1):

        self.infodicts = self._get_infodicts()

        if nprocess > len(self.infodicts):
            nprocess = len(self.infodicts)
        info_chunks = partition(self.infodicts, nprocess)

        pool = mp.Pool(processes=nprocess)
        try:
            pp = pool.map_async(_run_validation_chunks, info_chunks)
            # the results here should be a list of score values
            self.result = pp.get(86400)  # apparently this counters a bug in the exception passing in python.subprocess...
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
            pool.join()
        else:
            pool.close()
            pool.join()

        self.train_scores = []
        self.test_scores = []
        for res in self.result:
            if self.score_train:
                self.train_scores.append(res[0])
            self.test_scores.append(res[1])

        if self.score_train:
            self.train_scores = np.concatenate(self.train_scores).reshape((len(self.param_list), self.cv))
        self.test_scores = np.concatenate(self.test_scores).reshape((len(self.param_list), self.cv))


def _run_validation_chunks(infodicts):

    train_scores = []
    test_scores = []

    train_allscores = []
    test_allscores = []
    try:
        for info in infodicts:
            kdes = KDEScorer(info)
            kdes.train()
            if info["score_train"]:
                train_score, tmp = kdes.score(on="train")
                train_scores.append(train_score)
                train_allscores.append(tmp)
            #
            test_score, tmp = kdes.score(on="test")
            test_scores.append(test_score)
            test_allscores.append(tmp)

    except KeyboardInterrupt:
            pass

    return train_scores, test_scores, (train_allscores, test_allscores)
    # return


INFVAL = -16  # this is the value we replace -inf with, effectively a surrogate for log(0), an obvious approximation...
def force_finite(arr):
    inds = np.invert(np.isfinite(arr))
    arr[inds] = INFVAL
    return arr


def ring_area(r1, r2):
    val = np.pi * (r2**2. - r1**2.)
    return val


class KDEScorer(object):
    def __init__(self, info):
        # self.xarr = from_buffer(info["mp_xarr"], info["mp_xarr_shape"])
        # self.weights = from_buffer(info["mp_w"], info["mp_w_shape"])
        self.xarr = info["xarr"]
        self.weights = info["w"]

        self.train_inds = info["split"][0]
        self.test_inds = info["split"][1]

        self.params = info["params"]
        self.params.update(info["extra_params"])

        self.kde = neighbors.KernelDensity(**self.params)

    def train(self):
        self.kde.fit(self.xarr.values[self.train_inds, :], sample_weight=self.weights.values[self.train_inds])

    def score(self, on="test"):
        if on == "test":
            index = self.test_inds
        elif on == "train":
            index = self.train_inds
        else:
            raise KeyError

        raw_scores = self.kde.score_samples(self.xarr.values[index, :])
        scores = force_finite(raw_scores)

        # this does not inlcude weights, but is OK as we don't want result skewed by the high weight limb...
        res = scores.sum() / len(scores)

        return res, scores


def get_nearest(val, arr):
    res = []
    for tmp in val:
        res.append(np.argmin((tmp - arr)**2.))
    return np.array(res)


class CompositeDraw(object):
    def __init__(self, wemu, demu, ipivot=22.4, whistsize=1e6, icutmin=22, chunksize=1e5, rng=None):
        self.wemu = wemu
        self.demu = demu
        self.ipivot = ipivot
        self.whistsize = whistsize
        self.icutmin = icutmin
        self.chunksize = int(chunksize)

        self._mkref()
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

    def _mkref(self):
        self.ibins = np.linspace(13, 30, 500)
        self.icens = self.ibins[:-1] + np.diff(self.ibins) / 2.

        wsamples = self.wemu.draw(self.whistsize)

        self.wip = np.histogram(wsamples.data["MAG_I"], weights=wsamples.weights, bins=self.ibins, density=True)[0]
        self.dip = np.histogram(self.demu.container.data["MAG_I"], weights=self.demu.container.weights, bins=self.ibins, density=True)[0]

        iscale = np.argmin((self.icens - self.ipivot) ** 2.)
        self.ifactor = self.wip[iscale] / self.dip[iscale]
        self.refcurve = self.dip * self.ifactor

        self.dip0 = np.zeros(len(self.icens))
        ii = self.icens <= self.ipivot
        self.dip0[ii] = self.wip[ii]
        ii = self.icens > self.ipivot
        self.dip0[ii] = self.dip[ii] * self.ifactor

        self.fracdeep = np.sum(self.dip0) / np.sum(self.wip) - 1.

    def draw(self, wide_samples_to_draw):

        self.wide_samples_to_draw = int(wide_samples_to_draw)
        self.deep_samples_to_draw = int(self.wide_samples_to_draw * self.fracdeep)


        wide = self.wemu.draw(self.wide_samples_to_draw)
        wide.set_mode("data")
        deep = self._deepdraw()
        deep.set_mode("data")

        joint = DualContainer()
        joint.set_data(pd.concat((wide.data, deep.data), sort=False).reset_index(drop=True))
        joint.set_mode("data")

        return joint, wide, deep

    def _deepdraw(self):

        chunks = []
        nobjs = 0
        while nobjs < self.deep_samples_to_draw:

            dsamples = self.demu.draw(self.chunksize)
            dsamples.set_mode("data")
            rands = self.rng.uniform(size=len(dsamples.data))
            inodes = get_nearest(dsamples["MAG_I"], self.icens)

            refvals = self.refcurve[inodes]
            wvals = self.wip[inodes]
            bools = (rands * refvals > wvals) & (dsamples["MAG_I"] > self.icutmin)

            chunks.append(dsamples.data[bools])
            nobjs += bools.sum()
        chunks = pd.concat(chunks)
        chunks = chunks.iloc[:self.deep_samples_to_draw].copy()

        tmp = self.wemu.draw(self.deep_samples_to_draw)
        chunks["LOGR"] = tmp.data["LOGR"].values
        chunks = chunks.reset_index(drop=True)
        cols = [chunks.columns[-1], ] + list(chunks.columns[:-1])
        # print(cols)
        chunks = chunks[cols]

        res = DualContainer()
        res.set_data(chunks)
        return res




class PowerKDE(object): # FIXME this should be inhaerited from a sklearn class
    def __init__(self):
        """This class should have a """
        pass


    def score(self):
        """This should be the radially balanced version of the normal score, with the notation that the first column is always radius..."""
        pass





##############################################################



# class ConstructMock(object):
#     """TODO Construct a detailed Line-of-Sight based on the emulated information"""
#     def __init__(self, bcg_feature, gal_feature, rng=None):
#         self.bcg_feature = bcg_feature
#         self.gal_feature = gal_feature
#
#         if rng is None:
#             self.rng = np.random.RandomState()
#         else:
#             self.rng = rng
#
#     def train(self):
#         self.bcg_kde = FeatureEmulator(self.bcg_feature, rng=self.rng)
#         self.bcg_kde.train()
#
#         self.gal_kde = FeatureEmulator(self.gal_feature, rng=self.rng)
#         self.gal_kde.train()
#
#     def create_table(self, mag_to_flux):
#         """
#         Create a rectangular image with Fluxes,
#         RA, DEC, Flux (g, r, i, z), size, |g|,
#
#         add metadata for the details of what
#
#         Write it to Pandas DataFrame to HDF5 file
#         """
#
#         # draw random profile
#         gal_num = self.gal_feature.abs_profile.sum()
#         gals = self.gal_kde.draw(gal_num, expand=True, linear=True)
#         gals = pd.DataFrame(data=gals, columns=self.gal_feature.features.columns)
#
#         bcg = self.bcg_kde.draw(1, expand=True, linear=True)
#         bcg = pd.DataFrame(data=bcg, columns=self.bcg_feature.features.columns)
#         bcg["DIST"] = 0
#
#         self.mock = pd.concat((bcg, gals))
#         self.mock = self.draw_pos(self.mock)
#         self.mock = self.add_flux(self.mock, mag_to_flux)
#
#     def draw_pos(self, mock):
#         """Adds RA DEC position to mock data"""
#         angles = self.rng.uniform(0, 2 * np.pi, len(mock))
#         mock["RA"] = mock["DIST"] * np.cos(angles)
#         mock["DEC"] = mock["DIST"] * np.sin(angles)
#         return mock
#
#     def add_flux(self, mock, mag_to_flux):
#         for i, row in enumerate(mag_to_flux):
#             mag = mock[row[1][0]].copy()
#             if row[2] == "+":
#                 for val in row[1][1:]:
#                     mag += mock[val]
#             elif row[2] == "-":
#                 for val in row[1][1:]:
#                     mag -= mock[val]
#
#             mock[row[0]] = 10.**((30. - mag) / 2.5)
#         return mock
#
#     def write_mock(self, fname):
#         self.mock.to_hdf(fname, key="data")
#
#

class RejectionSampler(object):
    def __init__(self):
        pass


class DistanceClassifier(object):
    """TODO Classify objects into cluster or field redshift based on priors"""
    def __init__(self):
        pass

    def classify(self):
        pass

    def write(self):
        pass

    def load(self):
        pass


class ConstructLOS(object):
    """TODO Construct a detailed Line-of-Sight based on the emulated information"""
    def __init__(self):
        pass

    def draw(self):
        pass

    def write(self):
        pass

    def load(self):
        pass

    def create_lostable(self):
        pass