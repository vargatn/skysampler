"""
Should containe the Gaussian Processes operations


In addition to the feature spaces we should also take into account the average numbers of objects,
e.g. radial number profile (in absolute terms)


"""

import numpy as np
import pandas as pd
import sklearn.neighbors as neighbors

_DEFAULT_BANDWIDTH = 0.06 # this is just an educated guess...


class FeatureSpaceContainer(object):
    def __init__(self, indexed_survey):
        """
        Container for Feature space (should be very fine histogram, to reduce memory size

        We will use it for later processing, and for this reason should be completely self-standing, and not memory heavy

        Parameters
        ----------
        indexed_survey
        """
        self.indexed_survey = indexed_survey
        self.samples = indexed_survey.samples

        self.get_weights()
        self.alldata = pd.concat(self.samples)
        self.weights = self.alldata["WEIGHT"]

        self.nobj = len(self.indexed_survey.target["inds"][0])

    def get_weights(self):
        self.rbin_weights = self.indexed_survey.numprof / self.indexed_survey.samples_nrows
        for d in np.arange(len(self.indexed_survey.rcens) + 2):
            self.samples[d]["WEIGHT"] = self.rbin_weights[d]

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

        self._rescale()
        self.outer_radial_profile()

    def _rescale(self):
        self.means = self.features.mean(axis=0)
        self.sigma = self.features.std(axis=0)
        self.xarr = ((self.features - self.means) / self.sigma).values

    def outer_radial_profile(self, scaler=10):
        """Calculates radial mean number profile around targets (only for the log-range part)"""
        self.dens_profile = self.indexed_survey.numprof[2:] / self.indexed_survey.rareas / self.nobj * scaler
        self.dens_err = np.sqrt(self.indexed_survey.numprof[2:]) / self.indexed_survey.rareas / self.nobj * scaler
        self.abs_profile = self.indexed_survey.numprof[2:] / self.nobj * scaler


class FeatureEmulator(object):
    def __init__(self, feature, rng=None):
        """Emulator for feature space"""
        self.feature = feature
        self.kde = None

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    def train(self, bandwidth=_DEFAULT_BANDWIDTH):
        """train the emulator"""
        self.bandwidth = bandwidth
        self.kde = neighbors.KernelDensity(bandwidth=self.bandwidth)
        self.kde.fit(self.feature.xarr, sample_weight=self.feature.weights)

    def draw(self, num, expand=True, linear=True):
        """draws random samples from KDE"""
        res = self.kde.sample(n_samples=int(num))
        if expand:
            res = res * self.feature.sigma.values + self.feature.means.values
            if linear:
                for i, log in enumerate(self.feature.logs):
                    if log:
                        res[:, i] = 10**res[:, i]
                    # print(i)
        return res


class ConstructMock(object):
    """TODO Construct a detailed Line-of-Sight based on the emulated information"""
    def __init__(self, bcg_feature, gal_feature, rng=None):
        self.bcg_feature = bcg_feature
        self.gal_feature = gal_feature

        if rng is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

    def train(self):
        self.bcg_kde = FeatureEmulator(self.bcg_feature)
        self.bcg_kde.train()

        self.gal_kde = FeatureEmulator(self.gal_feature)
        self.gal_kde.train()

    def create_table(self, mag_to_flux):
        """
        Create a rectangular image with Fluxes,
        RA, DEC, Flux (g, r, i, z), size, |g|,

        add metadata for the details of what

        Write it to Pandas DataFrame to HDF5 file
        """

        # draw random profile
        gal_num = self.gal_feature.abs_profile.sum()
        gals = self.gal_kde.draw(gal_num, expand=True, linear=True)
        gals = pd.DataFrame(data=gals, columns=self.gal_feature.features.columns)

        bcg = self.bcg_kde.draw(1, expand=True, linear=True)
        bcg = pd.DataFrame(data=bcg, columns=self.bcg_feature.features.columns)
        bcg["DIST"] = 0

        self.mock = pd.concat((bcg, gals))
        self.mock = self.draw_pos(self.mock)
        self.mock = self.add_flux(self.mock, mag_to_flux)

    def draw_pos(self, mock):
        """Adds RA DEC position to mock data"""
        angles = self.rng.uniform(0, 2 * np.pi, len(mock))
        mock["RA"] = mock["DIST"] * np.cos(angles)
        mock["DEC"] = mock["DIST"] * np.sin(angles)
        return mock

    def add_flux(self, mock, mag_to_flux):
        for i, row in enumerate(mag_to_flux):
            mag = mock[row[1][0]].copy()
            if row[2] == "+":
                for val in row[1][1:]:
                    mag += mock[val]
            elif row[2] == "-":
                for val in row[1][1:]:
                    mag -= mock[val]

            mock[row[0]] = 10.**((30. - mag) / 2.5)
        return mock

    def write_mock(self, fname):
        self.mock.to_hdf(fname, key="data")



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