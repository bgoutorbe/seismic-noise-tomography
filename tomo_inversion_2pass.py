"""
This script performs a two-pass tomographic inversion of a set of
observed travel-times (equivalent to observed velocities) between
pairs of stations, at various periods.

The script takes as input one or several binary files containing a
list of dispersion curves (previously calculated with, e.g., script
dispersion_curves.py). A dispersion curve is an instance of
pstomo.DispersionCurve exported in binary format with module pickle.

In the first pass, an overdamped tomographic inversion is
performed, and the relative differences between observed/
predicted travel-time are estimated for each observed
travel-time (i.e., for each pair of stations).

A second tomographic inversion is then performed, after
rejecting the pairs whose observed travel-time is too
different from the predidcted travel-time. The difference
threshold is given in *MAX_TRAVELTIME_RELDIFF*.

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

# periods
PERIODS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

# max relative diff between observed/predicted travel-time
# to keep pair in the second pass
MAX_TRAVELTIME_RELDIFF = 0.15

# parameters for the 1st and 2nd pass, respectively
GRID_STEPS = (1.0, 1.0)
MINPECTSNRS = (7.0, 7.0)
CORR_LENGTHS = (150, 150)
ALPHAS = (3000, 600)
BETAS = (200, 200)
LAMBDAS = (0.3, 0.3)

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

    # opening pdf file (setting name as "2-pass-tomography_xxx.pdf")
    try:
        os.makedirs(TOMO_DIR)
    except:
        pass
    basename = os.path.basename(pickle_file).replace('FTAN', '2-pass-tomography')
    pdfname = os.path.join(TOMO_DIR, os.path.splitext(basename)[0]) + '.pdf'
    print "Maps will be exported to pdf file: " + pdfname
    if os.path.exists(pdfname):
        # backup
        shutil.copyfile(pdfname, pdfname + '~')
    pdf = PdfPages(pdfname)

    # performing tomographic inversions at given periods
    for period in PERIODS:
        print "\nDoing period = {} s".format(period)

        # 2-pass inversion
        skippairs = []
        for passnb in (0, 1):
            s = ("{} pass (rejecting {} pairs): grid step = {}, min SNR = {}, "
                 "corr. length = {} km, alpha = {}, beta = {}, lambda = {}")
            print s.format('1st' if passnb == 0 else '2nd', len(skippairs),
                           GRID_STEPS[passnb], MINPECTSNRS[passnb],
                           CORR_LENGTHS[passnb], ALPHAS[passnb],
                           BETAS[passnb], LAMBDAS[passnb])

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
                                   skippairs=skippairs,
                                   verbose=False,
                                   lonstep=GRID_STEPS[passnb],
                                   latstep=GRID_STEPS[passnb],
                                   minspectSNR=MINPECTSNRS[passnb],
                                   correlation_length=CORR_LENGTHS[passnb],
                                   alpha=ALPHAS[passnb],
                                   beta=BETAS[passnb],
                                   lambda_=LAMBDAS[passnb])

            # creating a figure summing up the results of the inversion:
            # - 1st panel = map of velocities or velocity anomalies
            # - 2nd panel = map of interstation paths and path densities
            #               (highlighting paths with large diff
            #                between obs/predicted travel-time)
            # - 3rd panel = resolution map
            #
            # See doc of VelocityMap.plot(), VelocityMap.plot_velocity(),
            # VelocityMap.plot_pathdensity(), VelocityMap.plot_resolution()
            # for a detailed description of the input arguments.

            title = ("Period = {0} s, {1} pass, grid {2} x {2} deg, "
                     "min SNR = {3}, corr. length = {4} km, alpha = {5}, "
                     "beta = {6}, lambda = {7} ({8} paths, {9} rejected)")
            title = title.format(period, '1st' if passnb == 0 else '2nd',
                                 GRID_STEPS[passnb], MINPECTSNRS[passnb],
                                 CORR_LENGTHS[passnb], ALPHAS[passnb],
                                 BETAS[passnb], LAMBDAS[passnb], len(v.paths),
                                 len(skippairs))
            fig = v.plot(title=title, showplot=False,
                         terr_threshold=MAX_TRAVELTIME_RELDIFF)

            # exporting plot to pdf
            pdf.savefig(fig)
            plt.close()

            # pairs to reject (because observed/predicted travel-time is too large)
            if passnb == 0:
                terrs = v.traveltime_reldiffs()
                skippairs = [(c.station1.name, c.station2.name)
                             for c, terr in zip(v.disp_curves, terrs)
                             if abs(float(terr)) > MAX_TRAVELTIME_RELDIFF]

        # let's compare the 2-pass tomography with a one-pass tomography
        s = ("One-pass tomography: grid step = {}, min SNR = {}, "
             "corr. length = {} km, alpha = {}, beta = {}, lambda = {}")
        print s.format(GRID_STEPS[1], MINPECTSNRS[1], CORR_LENGTHS[1],
                       ALPHAS[1], BETAS[1], LAMBDAS[1])

        # tomographic inversion
        v = pstomo.VelocityMap(dispersion_curves=curves,
                               period=period,
                               verbose=False,
                               lonstep=GRID_STEPS[1],
                               latstep=GRID_STEPS[1],
                               minspectSNR=MINPECTSNRS[1],
                               correlation_length=CORR_LENGTHS[1],
                               alpha=ALPHAS[1],
                               beta=BETAS[1],
                               lambda_=LAMBDAS[1])

        # figure (highlighting paths with large diff
        # between obs/predicted travel-time)
        title = ("Period = {0} s, one pass, grid {1} x {1} deg, "
                 "min SNR = {2}, corr. length = {3} km, alpha = {4}, "
                 "beta = {5}, lambda = {6} ({7} paths)")
        title = title.format(period, GRID_STEPS[1], MINPECTSNRS[1],
                             CORR_LENGTHS[1], ALPHAS[1],
                             BETAS[1], LAMBDAS[1], len(v.paths))
        fig = v.plot(title=title, showplot=False,
                     terr_threshold=MAX_TRAVELTIME_RELDIFF)

        # exporting plot in pdf
        pdf.savefig(fig)
        plt.close()

    # closing pdf file
    pdf.close()

    # merging pages of pdf with similar period
    pagenbs = range(len(PERIODS) * 3)  # 3 figures per period (two-pass + one-pass)
    key = lambda pagenb: int(pagenb / 3)  # grouping pages 0-1-2, then 3-4-5 etc.

    pagesgroups = psutils.groupbykey(pagenbs, key=key)
    print "\nMerging pages of pdf..."
    psutils.combine_pdf_pages(pdfname, pagesgroups, verbose=True)