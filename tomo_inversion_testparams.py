"""
This script performs a tomographic inversions of a set of
velocities between pairs of stations, systematically varying
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
MINPECTSNRS = [10.0, 8.0, 7.0]

# parsing configuration file to import dirs
from pysismo.psconfig import FTAN_DIR, TOMO_DIR

# selecting dispersion curves
flist = sorted(glob.glob(os.path.join(FTAN_DIR, 'FTAN*.pickle*')))
print 'Select file containing dispersion curves:'
print '0 - All'
print '\n'.join('{} - {}'.format(i + 1, os.path.basename(f)) for i, f in enumerate(flist))
res = int(raw_input('\n'))
pickle_files = flist if res == 0 else [flist[res - 1]]

# loop on pickled curves
for pickle_file in pickle_files:
    print "\nProcessing dispersion curve in file: " + pickle_file

    f = open(pickle_file, 'rb')
    curves = pickle.load(f)
    f.close()

    # opening pdf file
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
    params_product = list(it.product(PERIODS, GRID_STEPS, MINPECTSNRS))
    for period, grid_step, minspectSNR in params_product:
        s = "Period = {} s, grid step = {}, min SNR = {}"
        print s.format(period, grid_step, minspectSNR)

        # velocity map at period, with given parameters
        v = pstomo.VelocityMap(dispersion_curves=curves, period=period,
                               lonstep=grid_step, latstep=grid_step,
                               minspectSNR=minspectSNR, verbose=False)

        # figure
        title = "Period = {0} s, grid {1} x {1} deg, min SNR = {2} ({3} paths)"
        title = title.format(period, grid_step, minspectSNR, len(v.paths))
        fig = v.plot(title=title, showplot=False)

        # exporting plot in pdf
        pdf.savefig(fig)
        plt.close()

    # closing pdf file
    pdf.close()

    # merging pages of pdf with similar period
    pagenbs = range(len(params_product))

    def key(pagenb):
        period, _, _ = params_product[pagenb]
        return period

    pagesgroups = psutils.groupbykey(pagenbs, key=key)
    print "\nMerging pages of pdf..."
    psutils.combine_pdf_pages(pdfname, pagesgroups, verbose=True)
