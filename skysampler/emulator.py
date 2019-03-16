"""
Should containe the Gaussian Processes operations


In addition to the feature spaces we should also take into account the average numbers of objects,
e.g. radial number profile (in absolute terms)


"""

import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors


def get_angle(num, rng):
    angle = rng.uniform(0, np.pi, size=num)
    return angle


def weighted_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, axis=0, weights=weights)
    variance = np.average((values-average)**2, axis=0, weights=weights)  # Fast and numerically precise
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

    def __init__(self, fsc, radial_splits=None):
        """
        Constructs a large set of comparison images

        Density comparisons  and such

        """
        self.fsc = fsc

        if radial_splits is not None:
            self._radial_splits = radial_splits

    def _get_col(self, label):
        if label in self.fsc.features.columns:
            col = self.fsc.features[label]
        elif label in self._reconstr_mags:
            tokens = self._reconstr_mags[label].split()
            col = self.fsc.features[tokens[0]]
            for i in np.arange(len(tokens) // 2):
                col = _OPERATORS[tokens[i + 1]](col, self.fsc.features[tokens[i + 2]])
        else:
            raise KeyError

        return col

    def radial_series(self, label1="MAG_I", label2="COLOR_R_I", rlabel="LOGR",
                      rlog=True, bins=None, fname=None, vmax=None, nbins=60):

        rr = 10 ** self.fsc.features[rlabel]

        col1 = self._get_col(label1)
        col2 = self._get_col(label2)

        ww = self.fsc.weights

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

        for axs in axarr:
            axs[0].set_ylabel(label2)

        for axs in axarr[-1]:
            axs.set_xlabel(label1)

        for i, ax in enumerate(faxarr):
            ind = (rr > self._radial_splits[i]) & (rr < self._radial_splits[i + 1])
            if i == 0 and vmax is None:
                tmp = np.histogram2d(col1[ind], col2[ind], weights=ww[ind], bins=bins, normed=True)[0]
                vmax = 1 * tmp.max()
            ax.hist2d(col1[ind], col2[ind], weights=ww[ind], bins=bins,
                      cmap=plt.cm.terrain, norm=mpl.colors.LogNorm(), normed=True, vmax=vmax)
            ax.grid(ls=":")

            ax.text(0.05, 0.87,
                    "$R\in[{:.2f};{:.2f})$".format(self._radial_splits[i], self._radial_splits[i + 1]),
                    transform=ax.transAxes)

        if fname is not None:
            fig.savefig(fname, dpi=300, bbox_inches="tight")
        return fig, axarr

    def set_info(self, radial_splits=None, cm_diagrams=None, cc_diagrams=None, plot_series=None,
                 reconstr_mags=None, mag_bins=None, color_bins=None):

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

    def corner(self, rbin=None, rlabel="LOGR", rlog=True, clognorm=True, nbins=60,
               fname=None, vmax=None):
        # This should be one radial bin, or if None, then all

        rr = 10 ** self.fsc.features[rlabel]

        if rbin is not None:
            rind = (rr > self._radial_splits[rbin]) & (rr < self._radial_splits[rbin + 1])
        else:
            rind = np.ones(len(rr), dtype=bool)

        columns = list(self.fsc.features.columns)

        bins = []
        for col in columns:
            bins.append(np.linspace(self.fsc.features[col].min(),
                                    self.fsc.features[col].max(), nbins))

        nrows = len(columns)
        ncols = len(columns)
        xsize = 2.5 * nrows
        ysize = 2.5 * ncols

        fig, axarr = plt.subplots(nrows=nrows, ncols=nrows, figsize=(xsize, ysize))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        faxarr = axarr.flatten()

        # blocking upper triangle
        for i in np.arange(nrows):
            for j in np.arange(ncols):
                if j < i:
                    norm = mpl.colors.Normalize()
                    if clognorm:
                        norm = mpl.colors.LogNorm()

                    axarr[i, j].hist2d(self.fsc.features[columns[j]][rind],
                                       self.fsc.features[columns[i]][rind],
                                       weights=self.fsc.weights[rind],
                                       bins=(bins[j], bins[i]), cmap=plt.cm.terrain,
                                       norm=norm, normed=True, vmax=vmax)
                    axarr[i, j].grid(ls=":")
                if i == j:
                    axarr[i, j].hist(self.fsc.features[columns[i]][rind], bins=bins[i],
                                     weights=self.fsc.weights[rind],
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

    def plot_radial_ensemble(self, fname_root, cm=True, cc=True, radial=True, nbins=60, vmax=None):
        """loop through a set of predefined diagrams"""

        # c-m diagrams
        if cm:
            for cm in self._cm_diagrams:
                fname = fname_root + "_cm_" + cm[0] + "_" + cm[1] + ".png"
                print(fname)
                self.radial_series(label1=cm[0], label2=cm[1], fname=fname, nbins=nbins, vmax=vmax)

        # c-c diagrams
        if cc:
            for cc in self._cc_diagrams:
                fname = fname_root + "_cc_" + cc[0] + "_" + cc[1] + ".png"
                print(fname)
                self.radial_series(label1=cc[0], label2=cc[1], fname=fname, nbins=nbins, vmax=vmax)

                #         # all -radial bin
        if radial:
            fname = fname_root + "_corner_all.png"
            print(fname)
            self.corner(fname=fname, nbins=nbins, vmax=vmax)

            # radial bins
            for rbin in np.arange(len(self._radial_splits) - 1):
                fname = fname_root + "_corner_rbin{:02d}.png".format(rbin)
                print(fname)
                self.corner(rbin=rbin, fname=fname, nbins=nbins, vmax=vmax)


# This is just a standard function
class FeatureEmulator(object):
    def __init__(self):
        """This is just the packaged version of the KDE"""
        pass

    def cross_validate(self):
        pass


class PowerKDE(object): # FIXME this should be inherited from a sklearn class
    def __init__(self):
        """This class should have a """
        pass


    def score(self):
        """This should be the radially balanced version of the normal score, with the notation that the first column is always radius..."""
        pass





##############################################################




#

#
# _DEFAULT_BANDWIDTH = 0.06 # this is just an educated guess...
#
#

#
# class FeatureSpaceContainer(object):
#     def __init__(self, indexed_survey):
#         """
#         Container for Feature space (should be very fine histogram, to reduce memory size
#
#         We will use it for later processing, and for this reason should be completely self-standing, and not memory heavy
#
#         Parameters
#         ----------
#         indexed_survey
#         """
#         self.indexed_survey = indexed_survey
#         self.samples = indexed_survey.samples
#
#         self.get_weights()
#         self.alldata = pd.concat(self.samples)
#         self.weights = self.alldata["WEIGHT"]
#
#         self.nobj = len(self.indexed_survey.target["inds"][0])
#
#     def get_weights(self):
#         self.rbin_weights = self.indexed_survey.numprof / self.indexed_survey.samples_nrows
#         for d in np.arange(len(self.indexed_survey.rcens) + 2):
#             self.samples[d]["WEIGHT"] = self.rbin_weights[d]
#
#     def construct_features(self, columns, limits=None, logs=None):
#         self.columns = columns
#         self.limits = limits
#         self.logs = logs
#         self.features = pd.DataFrame()
#         self.inds = np.ones(len(self.alldata), dtype=bool)
#         for i, col in enumerate(columns):
#             if isinstance(col[1], str):
#                 self.features[col[0]] = self.alldata[col[1]]
#             else:
#                 if col[1][2] == "-":
#                     self.features[col[0]] = self.alldata[col[1][0]] - self.alldata[col[1][1]]
#                 elif col[1][2] == "+":
#                     self.features[col[0]] = self.alldata[col[1][0]] + self.alldata[col[1][1]]
#                 elif col[1][2] == "*":
#                     self.features[col[0]] = self.alldata[col[1][0]] * self.alldata[col[1][1]]
#                 elif col[1][2] == "/":
#                     self.features[col[0]] = self.alldata[col[1][0]] / self.alldata[col[1][1]]
#                 else:
#                     raise KeyError("only + - * / are supported at the moment")
#
#             if limits is not None:
#                 self.inds &= (self.features[col[0]] > limits[i][0]) & (self.features[col[0]] < limits[i][1])
#
#         self.features = self.features[self.inds]
#         self.weights = self.alldata["WEIGHT"][self.inds]
#
#         for i, col in enumerate(columns):
#             if logs is not None and logs[i]:
#               self.features[col[0]] = np.log10(self.features[col[0]])
#
#         self._rescale()
#         self.outer_radial_profile()
#
    # def _rescale(self):
    #     self.means = self.features.mean(axis=0)
    #     self.sigma = self.features.std(axis=0)
    #     self.xarr = ((self.features - self.means) / self.sigma).values
#
#     def outer_radial_profile(self, scaler=10):
#         """Calculates radial mean number profile around targets (only for the log-range part)"""
#         self.dens_profile = self.indexed_survey.numprof[2:] / self.indexed_survey.rareas / self.nobj * scaler
#         self.dens_err = np.sqrt(self.indexed_survey.numprof[2:]) / self.indexed_survey.rareas / self.nobj * scaler
#         self.abs_profile = self.indexed_survey.numprof[2:] / self.nobj * scaler
#
#
# class FeatureEmulator(object):
#     def __init__(self, feature, rng=None):
#         """Emulator for feature space"""
#         self.feature = feature
#         self.kde = None
#
#         if rng is None:
#             self.rng = np.random.RandomState()
#         else:
#             self.rng = rng
#
#     def train(self, bandwidth=_DEFAULT_BANDWIDTH):
#         """train the emulator"""
#         self.bandwidth = bandwidth
#         self.kde = neighbors.KernelDensity(bandwidth=self.bandwidth)
#         self.kde.fit(self.feature.xarr, sample_weight=self.feature.weights)
#
#     def draw(self, num, expand=True, linear=True):
#         """draws random samples from KDE"""
#         res = self.kde.sample(n_samples=int(num), random_state=self.rng)
#         if expand:
#             res = res * self.feature.sigma.values + self.feature.means.values
#             if linear:
#                 for i, log in enumerate(self.feature.logs):
#                     if log:
#                         res[:, i] = 10**res[:, i]
#         return res
#
#
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