"""
Module that parses global parameters from a configuration
file at first import, to make them available to the other
parts of the program.
"""

import ConfigParser
import os
import glob
import json
import datetime as dt
import numpy as np


def select_and_parse_config_file(basedir='.', ext='cnf', verbose=True):
    """
    Reads a configuration file and returns an instance of ConfigParser:

    First, looks for files in *basedir* with extension *ext*.
    Asks user to select a file if several files are found,
    and parses it using ConfigParser module.

    @rtype: L{ConfigParser.ConfigParser}
    """
    config_files = glob.glob(os.path.join(basedir, u'*.{}'.format(ext)))

    if not config_files:
        raise Exception("No configuration file found!")

    if len(config_files) == 1:
        # only one configuration file
        config_file = config_files[0]
    else:
        print "Select a configuration file:"
        for i, f in enumerate(config_files, start=1):
            print "{} - {}".format(i, f)
        res = int(raw_input(''))
        config_file = config_files[res - 1]

    if verbose:
        print "Reading configuration file: {}".format(config_file)

    conf = ConfigParser.ConfigParser()
    conf.read(config_file)

    return conf

# ==========================
# parsing configuration file
# ==========================

config = select_and_parse_config_file(basedir='.', ext='cnf', verbose=True)

# -----
# paths
# -----

# input dirs
MSEED_DIR = config.get('paths', 'MSEED_DIR')
STATIONXML_DIR = config.get('paths', 'STATIONXML_DIR')
DATALESS_DIR = config.get('paths', 'DATALESS_DIR')

# output dirs
CROSSCORR_DIR = config.get('paths', 'CROSSCORR_DIR')
FTAN_DIR = config.get('paths', 'FTAN_DIR')
TOMO_DIR = config.get('paths', 'TOMO_DIR')

# ---------------
# maps parameters
# ---------------

# paths to shapefiles (coasts, tectonic provinces and labels)
COAST_SHP = config.get('maps', 'COAST_SHP')
TECTO_SHP = config.get('maps', 'TECTO_SHP')
TECTO_LABELS = config.get('maps', 'TECTO_LABELS')

# colors of tectonic provinces
TECTO_COLORS = json.loads(config.get('maps', 'TECTO_COLORS'))

# bounding boxes
BBOX_LARGE = json.loads(config.get('maps', 'BBOX_LARGE'))
BBOX_SMALL = json.loads(config.get('maps', 'BBOX_SMALL'))

# --------------------------------------
# cross-correlation / spectra parameters
# --------------------------------------

# use dataless files or stationXML files to remove instrument response?
USE_DATALESSPAZ = config.getboolean('cross-correlation', 'USE_DATALESSPAZ')
USE_STATIONXML = config.getboolean('cross-correlation', 'USE_STATIONXML')

# subset of stations to cross-correlate
CROSSCORR_STATIONS_SUBSET = config.get('cross-correlation', 'CROSSCORR_STATIONS_SUBSET')
CROSSCORR_STATIONS_SUBSET = json.loads(CROSSCORR_STATIONS_SUBSET)

# locations to skip
CROSSCORR_SKIPLOCS = json.loads(config.get('cross-correlation', 'CROSSCORR_SKIPLOCS'))

# first and last day, minimum data fill per day
FIRSTDAY = config.get('cross-correlation', 'FIRSTDAY')
FIRSTDAY = dt.datetime.strptime(FIRSTDAY, '%d/%m/%Y').date()
LASTDAY = config.get('cross-correlation', 'LASTDAY')
LASTDAY = dt.datetime.strptime(LASTDAY, '%d/%m/%Y').date()
MINFILL = config.getfloat('cross-correlation', 'MINFILL')

# band-pass parameters
PERIODMIN = config.getfloat('cross-correlation', 'PERIODMIN')
PERIODMAX = config.getfloat('cross-correlation', 'PERIODMAX')
FREQMIN = 1.0 / PERIODMAX
FREQMAX = 1.0 / PERIODMIN
CORNERS = config.getint('cross-correlation', 'CORNERS')
ZEROPHASE = config.getboolean('cross-correlation', 'ZEROPHASE')
# resample period (to decimate traces, after band-pass)
PERIOD_RESAMPLE = config.getfloat('cross-correlation', 'PERIOD_RESAMPLE')

# Time-normalization parameters:
ONEBIT_NORM = config.getboolean('cross-correlation', 'ONEBIT_NORM')
# earthquakes period bands
PERIODMIN_EARTHQUAKE = config.getfloat('cross-correlation', 'PERIODMIN_EARTHQUAKE')
PERIODMAX_EARTHQUAKE = config.getfloat('cross-correlation', 'PERIODMAX_EARTHQUAKE')
FREQMIN_EARTHQUAKE = 1.0 / PERIODMAX_EARTHQUAKE
FREQMAX_EARTHQUAKE = 1.0 / PERIODMIN_EARTHQUAKE
# time window (s) to smooth data in earthquake band
# and calculate time-norm weights
WINDOW_TIME = 0.5 * PERIODMAX_EARTHQUAKE

# frequency window (Hz) to smooth ampl spectrum
# and calculate spect withening weights
WINDOW_FREQ = config.getfloat('cross-correlation', 'WINDOW_FREQ')

# Max time window (s) for cross-correlation
CROSSCORR_TMAX = config.getfloat('cross-correlation', 'CROSSCORR_TMAX')


# ---------------
# FTAN parameters
# ---------------

# default period bands, used to:
# - plot cross-correlation by period bands, in plot_FTAN(), plot_by_period_bands()
# - plot spectral SNR, in plot_spectral_SNR()
# - estimate min spectral SNR, in FTANs()
PERIOD_BANDS = json.loads(config.get('FTAN', 'PERIOD_BANDS'))

# default parameters to define the signal and noise windows used to
# estimate the SNR:
# - the signal window is defined according to a min and a max velocity as:
#   dist/vmax < t < dist/vmin
# - the noise window has a fixed size and starts after a fixed trailing
#   time from the end of the signal window

SIGNAL_WINDOW_VMIN = config.getfloat('FTAN', 'SIGNAL_WINDOW_VMIN')
SIGNAL_WINDOW_VMAX = config.getfloat('FTAN', 'SIGNAL_WINDOW_VMAX')
SIGNAL2NOISE_TRAIL = config.getfloat('FTAN', 'SIGNAL2NOISE_TRAIL')
NOISE_WINDOW_SIZE = config.getfloat('FTAN', 'NOISE_WINDOW_SIZE')

# smoothing parameter of FTAN analysis
FTAN_ALPHA = config.getfloat('FTAN', 'FTAN_ALPHA')

# periods and velocities of FTAN analysis
RAWFTAN_PERIODS_STARTSTOPSTEP = config.get('FTAN', 'RAWFTAN_PERIODS_STARTSTOPSTEP')
RAWFTAN_PERIODS_STARTSTOPSTEP = json.loads(RAWFTAN_PERIODS_STARTSTOPSTEP)
RAWFTAN_PERIODS = np.arange(*RAWFTAN_PERIODS_STARTSTOPSTEP)

CLEANFTAN_PERIODS_STARTSTOPSTEP = config.get('FTAN', 'CLEANFTAN_PERIODS_STARTSTOPSTEP')
CLEANFTAN_PERIODS_STARTSTOPSTEP = json.loads(CLEANFTAN_PERIODS_STARTSTOPSTEP)
CLEANFTAN_PERIODS = np.arange(*CLEANFTAN_PERIODS_STARTSTOPSTEP)

FTAN_VELOCITIES_STARTSTOPSTEP = config.get('FTAN', 'FTAN_VELOCITIES_STARTSTOPSTEP')
FTAN_VELOCITIES_STARTSTOPSTEP = json.loads(FTAN_VELOCITIES_STARTSTOPSTEP)
FTAN_VELOCITIES = np.arange(*FTAN_VELOCITIES_STARTSTOPSTEP)
FTAN_VELOCITIES_STEP = FTAN_VELOCITIES_STARTSTOPSTEP[2]

# relative strength of the smoothing term in the penalty function that
# the dispersion curve seeks to minimize
STRENGTH_SMOOTHING = config.getfloat('FTAN', 'STRENGTH_SMOOTHING')

# replace nominal frequancy (i.e., center frequency of Gaussian filters)
# with instantaneous frequency (i.e., dphi/dt(t=arrival time) with phi the
# phase of the filtered analytic signal), in the FTAN and dispersion curves?
# See Bensen et al. (2007) for technical details.
USE_INSTANTANEOUS_FREQ = config.getboolean('FTAN', 'USE_INSTANTANEOUS_FREQ')

# if the instantaneous frequency (or period) is used, we need to discard bad
# values from instantaneous periods. So:
# - instantaneous periods whose relative difference with respect to
#   nominal period is greater than ``MAX_RELDIFF_INST_NOMINAL_PERIOD``
#   are discarded,
# - instantaneous periods lower than ``MIN_INST_PERIOD`` are discarded,
# - instantaneous periods whose relative difference with respect to the
#   running median is greater than ``MAX_RELDIFF_INST_MEDIAN_PERIOD`` are
#   discarded; the running median is calculated over
#   ``HALFWINDOW_MEDIAN_PERIOD`` points to the right and to the left
#   of each period.

MAX_RELDIFF_INST_NOMINAL_PERIOD = config.getfloat('FTAN',
                                                  'MAX_RELDIFF_INST_NOMINAL_PERIOD')
MIN_INST_PERIOD = config.getfloat('FTAN', 'MIN_INST_PERIOD')
HALFWINDOW_MEDIAN_PERIOD = config.getint('FTAN', 'HALFWINDOW_MEDIAN_PERIOD')
MAX_RELDIFF_INST_MEDIAN_PERIOD = config.getfloat('FTAN',
                                                 'MAX_RELDIFF_INST_MEDIAN_PERIOD')

# --------------------------------
# Tomographic inversion parameters
# --------------------------------

# Default parameters related to the velocity selection criteria

# min spectral SNR to retain velocity
MINSPECTSNR = config.getfloat('tomography', 'MINSPECTSNR')
# min spectral SNR to retain velocity if no std dev
MINSPECTSNR_NOSDEV = config.getfloat('tomography', 'MINSPECTSNR_NOSDEV')
# max sdt dev (km/s) to retain velocity
MAXSDEV = config.getfloat('tomography', 'MAXSDEV')
# min nb of trimesters to estimate std dev
MINNBTRIMESTER = config.getint('tomography', 'MINNBTRIMESTER')
# max period = *MAXPERIOD_FACTOR* * pair distance
MAXPERIOD_FACTOR = config.getfloat('tomography', 'MAXPERIOD_FACTOR')

# Default internode spacing of grid
LONSTEP = config.getfloat('tomography', 'LONSTEP')
LATSTEP = config.getfloat('tomography', 'LATSTEP')

# Default correlation length of the smoothing kernel:
# S(r,r') = exp[-|r-r'|**2 / (2 * correlation_length**2)]
CORRELATION_LENGTH = config.getfloat('tomography', 'CORRELATION_LENGTH')

# Default strength of the spatial smoothing term (alpha) and the
# weighted norm penalization term (beta) in the penalty function
ALPHA = config.getfloat('tomography', 'ALPHA')
BETA = config.getfloat('tomography', 'BETA')

# Default parameter in the damping factor of the norm penalization term,
# such that the norm is weighted by exp(- lambda_*path_density)
# With a value of 0.15, penalization becomes strong when path density < ~20
# With a value of 0.30, penalization becomes strong when path density < ~10
LAMBDA = config.getfloat('tomography', 'LAMBDA')