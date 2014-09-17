"""
Performing raw-clean FTAN analysis to extract dispersion
curves (Rayleigh group velocity vs period) across pairs
of station.
"""

from pysismo import pscrosscorr
import glob
import os

# parsing configuration file to import dir of cross-corr results
from pysismo.psconfig import CROSSCORR_DIR

# loading cross-correlations
flist = sorted(glob.glob(os.path.join(CROSSCORR_DIR, 'xcorr*.pickle*')))
print 'Select file(s) containing cross-correlations to process: [All]'
print '0 - All'
print '\n'.join('{} - {}'.format(i + 1, os.path.basename(f))
                for i, f in enumerate(flist))
res = raw_input('\n')
pickle_files = flist if not res else [flist[int(i)-1] for i in res.split()]

# loop on cross-correlations
for pickle_file in pickle_files:
    print "\nProcessing cross-correlations of file: " + pickle_file
    xc = pscrosscorr.load_pickled_xcorr(pickle_file)

    # performing raw-clean FTAN, exporting FTANs to pdf and
    # dispersion curves to pickle file

    # copying suffix of cross-correlations file
    suffix = os.path.splitext(os.path.basename(pickle_file))[0].replace('xcorr_', '')
    xc.FTANs(suffix=suffix, whiten=True)