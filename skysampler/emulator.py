"""
Should containe the Gaussian Processes operations


In addition to the feature spaces we should also take into account the average numbers of objects,
e.g. radial number profile (in absolute terms)


"""

import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors
import matplotlib as mpl
try:
    import matplotlib.pyplot as plt
except:
    mpl.use("Agg")
    import matplotlib.pyplot as plt


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


class FeatureSpaceContainer(object):
    def __init__(self, info):
        """
        This needs to be done first
        """

        self.rcens = info.rcens
        self.redges = info.redges
        self.rareas = info.rareas

        self.survey = info.survey
        self.target = info.target

        self.numprof = info.numprof
        self.samples = info.samples

        self.alldata = pd.concat(self.samples).reset_index(drop=True)

        self.nobj = self.target.nrow

    def construct_features(self, columns, limits=None, logs=None):
        self.columns = columns
        self.limits = limits
        self.logs = logs
        self.features = pd.DataFrame()

        self.inds = np.ones(len(self.alldata), dtype=bool)
        for i, col in enumerate(columns):
            if isinstance(col[1], str):
                self.features[col[0]] = self.alldata[col[1]]
            else:
                if col[1][2] == "-":
                    self.features[col[0]] = self.alldata[col[1][0]] - self.alldata[col[1][1]]
                elif col[1][2] == "+":
                    self.features[col[0]] = self.alldata[col[1][0]] + self.alldata[col[1][1]]
                elif col[1][2] == "*":
                    self.features[col[0]] = self.alldata[col[1][0]] * self.alldata[col[1][1]]
                elif col[1][2] == "/":
                    self.features[col[0]] = self.alldata[col[1][0]] / self.alldata[col[1][1]]
                else:
                    raise KeyError("only + - * / are supported at the moment")

            if limits is not None:
                self.inds &= (self.features[col[0]] > limits[i][0]) & (self.features[col[0]] < limits[i][1])

        self.features = self.features[self.inds]
        self.weights = self.alldata["WEIGHT"][self.inds]

        for i, col in enumerate(columns):
            if logs is not None and logs[i]:
              self.features[col[0]] = np.log10(self.features[col[0]])

    def standardize(self):
        self.mean = np.average(self.features.values, axis=0, weights=self.weights)
        self.sigma = weighted_std(self.features, weights=self.weights)
        self.xarr = ((self.features - self.mean) / self.sigma)

    def rescale(self, tab):
        return (tab - self.mean) / self.sigma

    def inverse_rescale(self, tab):
        return tab * self.sigma + self.mean

    def surfdens(self, icol=0, scaler=1):
        if self.logs[icol]:
            arr = 10**self.features.values[:, icol]
        else:
            arr = self.features.values[:, icol]
        vals = np.histogram(arr, bins=self.redges, weights=self.weights)[0] / self.nobj / self.rareas * scaler
        return vals

    def to_dual(self):
        res = DualContainer(self.features.columns)
        res.set_data(self.features, self.weights)
        return res


class DualContainer(object):
    """Contains features in normal and in transformed space"""
    def __init__(self, columns=None, mean=None, sigma=None):
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

    def __getitem__(self, key):
        if self.mode == "xarr":
            return self.xarr[key]
        else:
            return self.data[key]

    def set_mode(self, mode):
        self.mode = mode

    def set_xarr(self, xarr):
        self.xarr = pd.DataFrame(columns=self.columns, data=xarr)

        self.data = self.xarr * self.sigma + self.mean
        self.weights = np.ones(len(self.xarr))
        self.shape = self.data.shape

    def set_data(self, data, weights=None):
        self.columns = data.columns
        self.data = data
        self.weights = weights

        if self.weights is None:
            self.weights = np.ones(len(self.data))

        self.mean = np.average(self.data, axis=0, weights=self.weights)
        self.sigma = weighted_std(self.data, weights=self.weights)

        self.xarr = ((self.data - self.mean) / self.sigma)
        self.shape = self.data.shape

    def transform(self, arr):
        return (arr - self.mean)/ self.sigma

    def inverse_transform(self, arr):
        return arr * self.sigma + self.mean


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

    def __init__(self, container, radial_splits=None):
        """
        Constructs a large set of comparison images

        Density comparisons  and such

        """
        self.container = container

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
            ax.hist2d(col1[ind], col2[ind], weights=ww[ind], bins=bins, cmap=plt.cm.viridis,
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

        if rlog:
            rr = 10 ** self.container[rlabel]
        else:
            rr = self.container[rlabel]

        if rbin is not None:
            rind = (rr > self._radial_splits[rbin]) & (rr < self._radial_splits[rbin + 1])
        else:
            rind = np.ones(len(rr), dtype=bool)

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
                                       bins=(bins[j], bins[i]), #cmap=plt.cm.terrain_r,
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

    def train(self, bandwidth=0.2, kernel="gaussian", **kwargs):
            """train the emulator"""
            self.bandwidth = bandwidth
            self.kde = neighbors.KernelDensity(bandwidth=self.bandwidth, kernel=kernel, **kwargs)
            self.kde.fit(self.container.xarr, sample_weight=self.container.weights)

    def draw(self, num, rmax=None, rcol="LOGR"):
        """draws random samples from KDE maximum radius"""
        res = DualContainer(columns=self.container.columns, mean=self.container.mean, sigma=self.container.sigma)
        _res = self.kde.sample(n_samples=int(num), random_state=self.rng)
        res.set_xarr(_res)
        if rmax is not None:
            inds = (res.data[rcol] > rmax)
            while inds.sum():
                print(inds.sum())
                vals = self.kde.sample(n_samples=int(inds.sum()), random_state=self.rng)
                _res[inds, :] = vals
                res.set_xarr(_res)
                inds = (res.data[rcol] > rmax)
        return res

    def score_samples(self, arr):
        """Assuming that arr is in the data format"""

        values = self.container.transform(arr)
        res = self.kde.score_samples(values)
        return res

def get_nearest(val, arr):
    res = []
    for tmp in val:
        res.append(np.argmin((tmp - arr)**2.))
    return np.array(res)

class CompositeDraw(object):
    def __init__(self, wemu, demu, ipivot=22.4, whistsize=1e6, icutmin=22, rng=None):
        self.wemu = wemu
        self.demu = demu
        self.ipivot = ipivot
        self.whistsize = whistsize
        self.icutmin = icutmin

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

        chunksize = 100000
        chunks = []
        nobjs = 0
        while nobjs < self.deep_samples_to_draw:

            dsamples = self.demu.draw(chunksize)
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


class Validator(object):
    """
    TODO validation package,

    Should automate splitting base data into training and test
    """
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