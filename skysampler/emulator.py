"""
Should containe the Gaussian Processes operations


In addition to the feature spaces we should also take into account the average numbers of objects,
e.g. radial number profile (in absolute terms)
"""


class ClusterFeatureEmulator(object):
    """
    Should take a feature space as an input along with some config, a

    Should be
    """

    def __init__(self):
        pass

    def train(self):
        """train the emulator"""
        pass

    def draw(self):
        """create a random realization with equal number density"""
        pass

    def write(self):
        pass

    def load(self):
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