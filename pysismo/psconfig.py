"""
Module that parses global parameters from a configuration
file at first import, to make them available to the other
parts of the program.
"""

from psutils import filelist
import ConfigParser
from obspy.core import UTCDateTime
import datetime as dt


def read_config_file(basedir='.', ext='cnf', verbose=True):
    """
    Reads a configuration file and returns an instance of ConfigParser:

    First, looks for files in *basedir* with extension *ext*.
    Asks user to select a file if several files are found,
    and parses it using ConfigParser module.

    @rtype: L{ConfigParser.ConfigParser}
    """
    config_files = filelist(basedir=basedir, ext=ext, subdirs=False)

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

config = read_config_file(basedir='.', ext='cnf', verbose=True)

# output dir for cross-correlation
CROSSCORR_DIR = config.get('paths', 'CROSSCORR_DIR')

# calc spectra (instead of cross-corr)?
CALC_SPECTRA = config.getboolean('cross-correlation', 'CALC_SPECTRA')

# use dataless files or stationXML files to remove instrument response?
USE_DATALESSPAZ = config.getboolean('cross-correlation', 'USE_DATALESSPAZ')
USE_STATIONXML = config.getboolean('cross-correlation', 'USE_STATIONXML')

# first and last day, minimum data fill per day
FIRSTDAY = config.get('cross-correlation', 'FIRSTDAY')
FIRSTDAY = UTCDateTime(dt.datetime.strptime(FIRSTDAY, '%d/%m/%Y'))
LASTDAY = config.get('cross-correlation', 'LASTDAY')
LASTDAY = UTCDateTime(dt.datetime.strptime(LASTDAY, '%d/%m/%Y'))
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
XCORR_TMAX = config.getfloat('cross-correlation', 'XCORR_TMAX')

# Parameters for spectrum calculation
SPECTRA_STATIONS = config.get('cross-correlation', 'SPECTRA_STATIONS').split(',')
SPECTRA_FIRSTDAY = config.get('cross-correlation', 'SPECTRA_FIRSTDAY')
SPECTRA_FIRSTDAY = UTCDateTime(dt.datetime.strptime(SPECTRA_FIRSTDAY, '%d/%m/%Y'))
SPECTRA_LASTDAY = config.get('cross-correlation', 'SPECTRA_LASTDAY')
SPECTRA_LASTDAY = UTCDateTime(dt.datetime.strptime(SPECTRA_LASTDAY, '%d/%m/%Y'))
# plot traces OF LAST DAY along with spectra?
PLOT_TRACES = config.getboolean('cross-correlation', 'PLOT_TRACES')