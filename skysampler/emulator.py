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
    """
    Container for Feature space (should be very fine histogram, to reduce memory size

    We will use it for later processing, and for this reason should be completely self-standing, and not memory heavy
    """
    def __init__(self, indexed_survey):
        self.indexed_survey = indexed_survey
        self.samples = indexed_survey.samples

        self.get_weights()
        self.alldata = pd.concat(self.samples)
        self.weights = self.alldata["WEIGHT"]

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
        nobj = len(self.indexed_survey.target["inds"][0])
        self.dens_profile = self.indexed_survey.numprof[2:] / self.indexed_survey.rareas / nobj * scaler
        self.dens_err = np.sqrt(self.indexed_survey.numprof[2:]) / self.indexed_survey.rareas / nobj * scaler
        self.abs_profile = self.indexed_survey.numprof[2:] / nobj * scaler


class ClusterFeatureEmulator(object):
    """
    Should take a feature space as an input along with some config, a

    Should be
    """

    def __init__(self, feature):
        self.feature = feature
        self.kde = None

    def train(self, bandwidth=_DEFAULT_BANDWIDTH):
        """train the emulator"""
        self.bandwidth = bandwidth
        self.kde = neighbors.KernelDensity(bandwidth=self.bandwidth)
        self.kde.fit(self.feature.xarr, sample_weight=self.feature.weights)

    def draw(self, num, expand=True):
        """create a random realization with equal number density"""
        res = self.kde.sample(n_samples=int(num))
        if expand:
            res = res * self.feature.sigma.values + self.feature.means.values
        return res

    def draw_radial(self, log=True, col=0, num=None):

        self.feature.outer_radial_profile()
        if num is None:
            num_to_draw = self.feature.abs_profile.sum()
        else:
            num_to_draw = num

        self.container = []
        vals = self.draw(num_to_draw, expand=True)
        rvals = vals[:, col]
        if log:
            rvals = 10**rvals

        counts, tmp = np.histogram(rvals, self.feature.indexed_survey.redges)
        return counts, vals



        # self.container = []
        # rvals = self.draw(num_to_draw, expand=True)[:, col]
        # if log:
        #     rvals = 10**rvals
        # digits = np.digitize(rvals, self.feature.indexed_survey.redges)
        #
        # nbins = len(prof_to_draw)
        # bins_to_test = np.arange(nbins) + 1
        #
        # for i, d in enumerate(bins_to_test):
        #     ind = np.where(digits == d)[0]
        #     print(i, prof_to_draw[i], len(ind))







    def _calc_prof(self, sample):
        pass






class Validator(object):
    """
    validation package,

    Should automate splitting base data into training and test
    """
    def __init__(self):
        pass


class DistanceClassifier(object):
    """Classify objects into cluster or field redshift based on priors"""
    def __init__(self):
        pass

    def classify(self):
        pass

    def write(self):
        pass

    def load(self):
        pass


class ConstructLOS(object):
    """Construct a detailed Line-of-Sight based on the emulated information"""
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