"""
This script performs a tomographic inversions of a set of
observed travel-times (equivalent to observed velocities)
between pairs of stations, systematically varying
the filtering and inversion parameters: period, grid size,
min SNR etc.

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
PERIODS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
GRID_STEPS = [1.0]
MINPECTSNRS = [7.0]
CORR_LENGTHS = [150]
ALPHAS = [600, 1500, 3000]
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

    # opening pdf file (setting name as "tomography_xxx.pdf")
    try:
        os.makedirs(TOMO_DIR)
    except:
        pass
    basename = os.path.basename(pickle_file).replace('FTAN', 'tomography')
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

        # velocity map at period, with given parameters
        v = pstomo.VelocityMap(dispersion_curves=curves, period=period, verbose=False,
                               lonstep=grid_step, latstep=grid_step,
                               minspectSNR=minspectSNR, correlation_length=corr_length,
                               alpha=alpha, beta=beta, lambda_=lambda_)

        # figure
        title = ("Period = {0} s, grid {1} x {1} deg, min SNR = {2}, corr. length "
                 "= {3} km, alpha = {4}, beta = {5}, lambda = {6} ({7} paths)")
        title = title.format(period, grid_step, minspectSNR, corr_length,
                             alpha, beta, lambda_, len(v.paths))
        fig = v.plot(title=title, showplot=False)

        # exporting plot in pdf
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
