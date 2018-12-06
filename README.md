
The SKYSAMPLER survey emulator
=================================================

**skysampler** is a python package which can draw random realizations of survey data, with the special aim
of creating random realizations of galaxy clusters and their surrounding galaxies.

The package:

* Handles wide field survey data to learn the joint feature distribution of detections and galaxies
* Draws mock realizations of cluster line-of-sights
* Provides interface for [GalSim](https://github.com/GalSim-developers/GalSim) to render the mock catalogs into full survey-like exposures

Generating mock observations takes place in a data driven way, i.e. clusters are constructed as they are seen in
the survey, not according to our theoretical models for them. Hence the products are not critically dependent
on our physical assumptions, only on survey conditions.


Contact
========

In case of questions or if you would like to use parts of this pipeline in a publication, please contact me at ::

    T.Varga [at] physik.lmu.de
