=====
OView
=====

OView is a multi-purpose image and data viewer for Omega hdf files.

The vision is to implement a core of reading and viewing functionality, and to
be extensible for supporting new Omega-60 or Omega-EP hdf data sources. Some
rudimentary data processing and analysis will be builtin, more in-depth
analysis projects may prefer to extend with subclassing or import.

**Supported file types**
------------------------

builtin support for the following file types:

- image files (.jpg, .png, .bmp, .tif)
- a subset of .hdf files from the OMEGA database:
  - p510
  - asbo
  - pxrdip

** Dependencies **
------------------

oview was last developed with the following environment:

- python 3.6 (earlier versions unverified)
- numpy
- scipy
- PyQt5
- Qt5
- qtpy
- pyqtgraph

Additional (optional) python packages for full functionality:

- matplotlib
- pyhdf

