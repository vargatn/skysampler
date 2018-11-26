
=============
API Reference
=============



Survey Indexing and Target Selection
------------------------------------

The two basic objects contain targets, and an interface for the underlying sky survey:

.. autosummary::
    :toctree: generated
    :template: custom.rst

    skysampler.indexer.TargetData
    skysampler.indexer.SurveyData

These can be used to define the indexer object:

.. autosummary::
    :toctree: generated
    :template: custom.rst

    skysampler.indexer.SurveyIndexer

Which upon completion returns a data container:

.. autosummary::
    :toctree: generated
    :template: custom.rst

    skysampler.indexer.IndexedDataContainer

Additional convenience functions:

.. autosummary::
    :toctree: generated

    skysampler.indexer.get_theta_edges
    skysampler.indexer.subsample
    skysampler.indexer.shuffle
    skysampler.indexer.get_ndraw



Cluster Emulator
----------------

add here

