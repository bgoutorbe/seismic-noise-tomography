"""
This script implements the two-step frequency-time analysis
(FTAN) on a set of cross-correlations, as described by Levshin &
Ritzwoller, "Automated detection, extraction, and measurement
of regional surface waves", Pure Appl. Geoph. (2001) and Bensen
et al., "Processing seismic ambient noise data to obtain
reliable broad-band surface wave dispersion measurements",
Geophys. J. Int. (2007).

In short, the envelope's amplitude of the (analytic representation
of the) cross-correlation is calculated and displayed after applying
narrow bandpass filters: this gives a 2D image, in which the x-axis
is the filter's center period and the y-axis is time (equivalent to
velocity). Tracking the time (equivalent to velocity) at which the
the amplitude reaches its maximum for each period gives the dispersion
curve, i.e., the group velocity function of period. This 'raw'
dispersion curve is used to set up a phase-matched filter, re-apply
a 'clean' FTAN and extract a 'clean' dispersion curve.

This script takes as input one or several binary files containing a
set of cross-correlations (previously calculated with, e.g., script
crosscorrelation.py). A set of cross-correlations is an instance of
pscrosscorr.CrossCorrelationCollection exported in binary format
with module pickle. Two file per set of cross-correlations are
produced:

- a pdf file illustrating the FTAN procedure on all cross-correlations:
  one page per cross-correlation, containing the original and
  band-passed cross-correlation, the amplitude and dispersion curve
  of the raw FTAN, of the clean FTAN, and a map locating the pair
  of stations.

- a binary file containing the clean dispersion curves exported
  with module pickle
"""

from pysismo import pscrosscorr
import glob
import os

# parsing configuration file to import dir of cross-corr results
from pysismo.psconfig import CROSSCORR_DIR

# loading cross-correlations (looking for *.pickle files in dir *CROSSCORR_DIR*)
flist = sorted(glob.glob(os.path.join(CROSSCORR_DIR, 'xcorr*.pickle*')))
print 'Select file(s) containing cross-correlations to process: [All except backups]'
print '0 - All except backups (*~)'
print '\n'.join('{} - {}'.format(i + 1, os.path.basename(f))
                for i, f in enumerate(flist))

res = raw_input('\n')
if not res:
    pickle_files = [f for f in flist if f[-1] != '~']
else:
    pickle_files = [flist[int(i)-1] for i in res.split()]

# processing each set of cross-correlations
for pickle_file in pickle_files:
    print "\nProcessing cross-correlations of file: " + pickle_file
    xc = pscrosscorr.load_pickled_xcorr(pickle_file)

    # copying the suffix of cross-correlations file
    # (everything between 'xcorr_' and the extension)
    suffix = os.path.splitext(os.path.basename(pickle_file))[0].replace('xcorr_', '')

    # Performing the two-step FTAN, exporting the figures to a
    # pdf file (one page per cross-correlation) and the clean
    # dispersion curves to a binary file using module pickle.
    #
    # The file are saved in dir *FTAN_DIR* (defined in configuration file) as:
    # <prefix>_<suffix>.pdf and <prefix>_<suffix>.pickle
    # You can specify *prefix* as input argument in FTANs(), or leave
    # the function define a default prefix, which will look like:
    # FTAN[_whitenedxc][_mindist=...][_minsSNR=...][_minspectSNR=...] ...
    # [_month-year_month-year]
    #
    # Set whiten=True to whiten the spectrum of the cross-correlations
    # (default is False)
    # Set normalize_ampl=True to normalize FTAN amplitude in plots, so
    # that the max amplitude = 1 at each period (default is True)
    # Set logscale=True to plot log(amplitude^2) instead of amplitude
    # (default is True)
    #
    # See other options in the docstring of the function.

    xc.FTANs(suffix=suffix, whiten=False, normalize_ampl=True, logscale=True)