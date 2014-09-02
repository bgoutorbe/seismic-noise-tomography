"""
This script performs a tomographic inversions of a set of
velocities between pairs of stations, systematically varying
the filtering and inversion parameters: period, grid size,
min SNR etc.

The results are exported in a pdf file in dir *TOMO_DIR*
"""

from pysismo import pstomo
import os
import shutil
import glob
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# constants and parameters
PERIODS = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
GRID_STEPS = [1.0, 0.5]
MINPECTSNRS = [10.0, 8.0, 7.0]

# parsing configuration file to import dirs
from pysismo.psconfig import FTAN_DIR, TOMO_DIR

# loading dispersion curves
flist = sorted(glob.glob(os.path.join(FTAN_DIR, 'FTAN*.pickle*')))
print 'Select file containing dispersion curves:'
print '\n'.join('{} - {}'.format(i, os.path.basename(f)) for i, f in enumerate(flist))
pickle_file = flist[int(raw_input('\n'))]
f = open(pickle_file, 'rb')
curves = pickle.load(f)
f.close()

# opening pdf file
try:
    os.makedirs(TOMO_DIR)
except:
    pass
basename = os.path.join(TOMO_DIR, os.path.splitext(os.path.basename(pickle_file))[0])
print "Exporting maps in file: {}.pdf".format(basename)
pdfname = basename + '.pdf'
if os.path.exists(pdfname):
    # backup
    shutil.copyfile(pdfname, pdfname + '~')
pdf = PdfPages(pdfname)

for period in PERIODS:
    print "Tomographic inversion at period = {} s".format(period)
    v = pstomo.VelocityMap(dispersion_curves=curves, period=period, verbose=False)
    fig = v.plot(showplot=False)

    # exporting plot in pdf
    pdf.savefig(fig)
    plt.close()

# closing pdf file
pdf.close()