.. skysampler documentation master file, created by
   sphinx-quickstart on Mon Nov 26 14:25:01 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================================
The SKYSAMPLER survey emulator
=================================================

**skysampler** is a python package which can draw random realizations of survey data, with the special aim
of creating random realizations of galaxy clusters and their surrounding galaxies.

The package:

* Handles wide field survey data to learn the joint feature distribution of detections and galaxies
* Draws mock realizations of cluster line-of-sights
* Provides interface for GalSim_ to render the mock catalogs into full survey-like exposures

.. _Galsim: https://github.com/GalSim-developers/GalSim

Generating mock observations takes place in a data driven way, i.e. clusters are constructed as they are seen in
the survey, not according to our theoretical models for them, hence the products are not critically dependent
on our physical assumptions, only on survey conditions.


.. toctree::
   :caption: User Documentation
   :maxdepth: 2

   API reference <api>


Contact
========

In case of questions or if you would like to use parts of this pipeline in a publication, please contact me at ::

    T.Varga [at] physik.lmu.de


