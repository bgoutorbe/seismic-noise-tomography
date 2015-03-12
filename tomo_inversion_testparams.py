#!/usr/bin/python -u
"""
[Advice: run this script using python with unbuffered output:
`python -u tomo_inversion_testparams.py`]

This script performs tomographic inversions of a set of
observed travel-times (equivalent to observed velocities)
between pairs of stations, systematically varying
the selection and inversion parameters: period, grid size,
min SNR etc.

The script takes as input one or several binary files containing a
list of dispersion curves (previously calculated with, e.g., script
dispersion_curves.py), located in folder *FTAN_DIR*. A dispersion
curve is an instance of pstomo.DispersionCurve exported in binary
format with module pickle.

The inversion is an implementation of the algorithm described
by Barmin et al., "A fast and reliable method for surface wave
tomography", Pure Appl. Geophys. (2001). The travel paths are
assumed to follow great circles between pairs of stations, so
that the relationship between the data (travel-time anomalies
between pairs of stations) and the parameters (slowness anomalies
at grid nodes) is linear. The penalty function is then composed
of three terms: the first represents the misfit between observed
and predicted data; the second is a spatial smoothing condition;
the third penalizes the weighted norm of the parameters:

- the spatial smoothing is controlled by a strength parameter,
  *alpha*, and a correlation length, *corr_length*;

- the norm penalization is controlled by a strength parameter,
  *beta*, and decreases as the path density increases, as
  exp[- *lambda* * path density]

Before the inversion is performed, several selection criteria
are applied to filter out low quality observed velocities.
The criteria are as follows:

1) period <= distance * *maxperiodfactor* (normally, distance / 12)
2) for velocities having a standard deviation associated:
   - standard deviation <= *maxsdev*
   - SNR >= *minspectSNR*
3) for velocities NOT having a standard deviation associated:
   - SNR >= *minspectSNR_nosdev*

The standard deviation of a velocity is estimated from the set
of trimester velocities (i.e., velocities estimated by performing
FTANs on cross-correlations calculated with 3 months of data,
Jan-Feb-Mar, Feb-Mar-Apr ... Dec-Jan-Feb) for which the SNR
is >= *minspectSNR*, and if at least *minnbtrimester* trimester
velocities are available.

The default value of all the parameters mentioned above is
defined in the configuration file, and can be overridden
when the inversion is performed, in pstomo.VelocityMap().

The results are exported in a pdf file in dir *TOMO_DIR*
"""

from pysismo import pstomo, psutils
import os
import shutil
import glob
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools as it

# inversion parameters to vary
PERIODS = [10.0, 20.0]
GRID_STEPS = [1.0]
MINPECTSNRS = [7.0]
CORR_LENGTHS = [50, 150, 250]
ALPHAS = [200, 400, 600]
BETAS = [200]
LAMBDAS = [0.3]

# parsing configuration file to import dirs
from pysismo.psconfig import FTAN_DIR, TOMO_DIR

# selecting dispersion curves
flist = sorted(glob.glob(os.path.join(FTAN_DIR, 'FTAN*.pickle*')))
print 'Select file(s) containing dispersion curves to process: [All except backups]'
print '0 - All except backups (*~)'
print '\n'.join('{} - {}'.format(i + 1, os.path.basename(f))
                for i, f in enumerate(flist))

res = raw_input('\n')
if not res:
    pickle_files = [f for f in flist if f[-1] != '~']
else:
    pickle_files = [flist[int(i)-1] for i in res.split()]

# loop on pickled curves
for pickle_file in pickle_files:
    print "\nProcessing dispersion curves of file: " + pickle_file

    f = open(pickle_file, 'rb')
    curves = pickle.load(f)
    f.close()

    # opening pdf file (setting name as "testparams-tomography_xxx.pdf")
    try:
        os.makedirs(TOMO_DIR)
    except:
        pass
    basename = os.path.basename(pickle_file).replace('FTAN', 'testparams-tomography')
    pdfname = os.path.join(TOMO_DIR, os.path.splitext(basename)[0]) + '.pdf'
    print "Maps will be exported to pdf file: " + pdfname
    if os.path.exists(pdfname):
        # backup
        shutil.copyfile(pdfname, pdfname + '~')
    pdf = PdfPages(pdfname)

    # performing tomographic inversions, systematically
    # varying the inversion parameters
    param_lists = it.product(PERIODS, GRID_STEPS, MINPECTSNRS, CORR_LENGTHS,
                             ALPHAS, BETAS, LAMBDAS)
    param_lists = list(param_lists)
    for period, grid_step, minspectSNR, corr_length, alpha, beta, lambda_ in param_lists:
        s = ("Period = {} s, grid step = {}, min SNR = {}, corr. length "
             "= {} km, alpha = {}, beta = {}, lambda = {}")
        print s.format(period, grid_step, minspectSNR, corr_length, alpha, beta, lambda_)

        # Performing the tomographic inversion to produce a velocity map
        # at period = *period* , with parameters given above:
        # - *lonstep*, *latstep* control the internode distance of the grid
        # - *minnbtrimester*, *maxsdev*, *minspectSNR*, *minspectSNR_nosdev*
        #   correspond to the selection criteria
        # - *alpha*, *corr_length* control the spatial smoothing term
        # - *beta*, *lambda_* control the weighted norm penalization term
        #
        # Note that if no value is given for some parameter, then the
        # inversion will use the default value defined in the configuration
        # file.
        #
        # (See doc of VelocityMap for a complete description of the input
        # arguments.)

        v = pstomo.VelocityMap(dispersion_curves=curves,
                               period=period,
                               verbose=False,
                               lonstep=grid_step,
                               latstep=grid_step,
                               minspectSNR=minspectSNR,
                               correlation_length=corr_length,
                               alpha=alpha,
                               beta=beta,
                               lambda_=lambda_)

        # creating a figure summing up the results of the inversion:
        # - 1st panel = map of velocities or velocity anomalies
        # - 2nd panel = map of interstation paths and path densities
        # - 3rd panel = resolution map
        #
        # See doc of VelocityMap.plot(), VelocityMap.plot_velocity(),
        # VelocityMap.plot_pathdensity(), VelocityMap.plot_resolution()
        # for a detailed description of the input arguments.

        title = ("Period = {0} s, grid {1} x {1} deg, min SNR = {2}, corr. length "
                 "= {3} km, alpha = {4}, beta = {5}, lambda = {6} ({7} paths)")
        title = title.format(period, grid_step, minspectSNR, corr_length,
                             alpha, beta, lambda_, len(v.paths))
        fig = v.plot(title=title, showplot=False)

        # exporting plot to pdf
        pdf.savefig(fig)
        plt.close()

    # closing pdf file
    pdf.close()

    # merging pages of pdf with similar period
    pagenbs = range(len(param_lists))

    def key(pagenb):
        period = param_lists[pagenb][0]
        return period

    pagesgroups = psutils.groupbykey(pagenbs, key=key)
    print "\nMerging pages of pdf..."
    psutils.combine_pdf_pages(pdfname, pagesgroups, verbose=True)
