Seismic noise tomography
========================
This project is dedicated to provide a Python framework for seismic noise tomography, 
based on [ObsPy](https://github.com/obspy/obspy/wiki) and numerical Python packages 
such as [numpy](http://www.numpy.org/) and [scipy](http://www.scipy.org/).

Requirements
------------
The code is developped and tested on Ubuntu (but should run on other platforms as well)
with Python 2.7.

In addition to [Python 2.7](https://www.python.org/download/releases/2.7/), you need
to install the following packages:

- [numpy](http://www.numpy.org/) >= 1.8.2
- [scipy](http://www.scipy.org/) >= 0.13.3
- [matplotlib](http://matplotlib.org/) >= 1.3.1
- [ObsPy](https://github.com/obspy/obspy/wiki) >= 0.9.2
- [pyshp](https://github.com/GeospatialPython/pyshp)
- [pyproj](https://code.google.com/p/pyproj/) >= 1.8.9
- [pyPdf](http://pybrary.net/pyPdf/)

It is recommended to install these packages with `pip install ...` or with your
favourite package manager, e.g., `apt-get install ...`.

How to start
------------
You should start reading the example configuration file, `tomo_Brazil.cnf`, which
contains global parameters and detailed instructions. You should then create 
your own configuration file (any name with cnf extension, \*.cnf) with your
own parameters, and place it in the same folder as the scripts. It is not advised
to simply modify `tomo_Brazil.cnf`, as any update may revert your changes.

You may then use the scripts in the following order:

- `crosscorrelation.py` takes seismic waveforms as input in order to calculate 
and export cross-correlations between pairs of stations,

- `dispersion_curves.py` takes cross-correlations as input and applies
a frequency-time analysis (FTAN) in order to extract and export group velocity
dispersion curves,

- `tomo_inversion_testparams.py` takes dispersion curves as input and applies
 a tomographic inversion to produce dispersion maps; the inversion parameters
 are systematically varied within user-defined ranges,

- `tomo_inversion_2pass.py` takes dispersion curves as input and applies
 a two-pass tomographic inversion to produce dispersion maps: an overdamped
 inversion is performed in the first pass in order to detect and reject outliers
 from the second pass.
 
The scripts rely on the Python package `pysismo`, which must thus be located
in a place included in your PATH (or PYTHONPATH) environment variable. The easiest
choice is of course to place it in the same folder as the scripts.

How to update
-------------
The code is still experimental so you should regularly check for (and pull) 
updates. These will be backward-compatible, **except if new parameters appear 
in the configuration file**.

**In other words, after any update, you should check whether new parameters were added
to the example configuration file (`tomo_Brazil.cnf`) and insert them accordingly
to your own configuration file.**