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

BADVAL = -9999.

class SurveyData(object):
    """
    Interface for Survey Data on disk


    Should know about file paths, and be able to loop through the files, reading them one by one
    """
    def __init__(self):
        pass

    def write(self):
        pass

    def load(self):
        pass


class SurveyIndexer(object):
    """
    Do the indexing of the Data based on passed cluster DataFrame


    This is mostly based on existing implementations

    What should be produced is the indexes of objects in radial shells, and their multiplicities.
    """
    def __init__(self):
        pass


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






