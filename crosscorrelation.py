#!/usr/bin/python -u
"""
[Advice: run this script using python with unbuffered output:
`python -u crosscorrelation.py`]

This script reads seismic waveform data from a set of stations, and
calculates the cross-correlations between all pairs of stations. The
data (in miniseed format) must be located in folder *MSEED_DIR*. The
stations information (coordinates, instrument response) can be read
from dataless seed files (if *USE_DATALESSPAZ* = True) located in
folder *DATALESS_DIR*, and/or stationXML files (if *USE_STATIONXML* =
True) located in folder *STATIONXML_DIR*. Note that two different
stations MUST HAVE DIFFERENT NAMES, even if they do not belong to
the same network. Also, one given station cannot have several
sets of coordinates: if so, it will be skipped.

In the current version of the program, miniseed files MUST be
organized inside their directory as:
<year>-<month>/<network>.<station>.<channel>.mseed, e.g.:
1988-10/BL.JFOB.BHZ.mseed
So, there is one sub-directory per month, and inside it, one miniseed
file  per month and per station.

The implemented algorithm follows the lines of Bensen et al.,
"Processing seismic ambient noise data to obtain reliable broad-band
surface wave dispersion measurements", Geophys. J. Int. (2007).

The procedure consists in stacking daily cross-correlations between
pairs of stations, from *FIRSTDAY* to *LASTDAY* and, in each given day,
rejecting stations whose data fill is < *MINFILL*. Define a subset of
stations to cross-correlate in *CROSSCORR_STATIONS_SUBSET* (or let it
empty to cross-correlate all stations). Define a list of locations to
skip in *CROSSCORR_SKIPLOCS*, if any. The cross-correlations are
calculated between -/+ *CROSSCORR_TMAX* seconds.

Several pre-processing steps are applied to the daily seismic waveform
data, before the daily cross-correlation is calculated and stacked:

(1) removal of the instrument response, the mean and the trend;

(2) band-pass filter between *PERIODMIN* and *PERIODMAX* sec

(3) down-sampling to sampling step = *PERIOD_RESAMPLE* sec

(4) time-normalization:

    - if *ONEBIT_NORM* = False, normalization of the signal by its
      (smoothed) absolute amplitude in the earthquake period band,
      defined as *PERIODMIN_EARTHQUAKE* - *PERIODMIN_EARTHQUAKE* sec.
      The smoothing window is *PERIODMAX_EARTHQUAKE* / 2;

    - if *ONEBIT_NORM* = False, one-bit normalization, wherein
      only the sign of the signal is kept (+1 or -1);

(5) spectral whitening of the Fourier amplitude spectrum: the Fourier
    amplitude spectrum of the signal is divided by a smoothed version
    of itself. The smoonthing window is *WINDOW_FREQ*.

Note that all the parameters mentioned above are defined in the
configuration file.

When all the cross-correlations are calculated, the script exports
several files in dir *CROSSCORR_DIR*, whose name (without extension)
is:

xcorr[_<stations of subset>]_<first year>-<last year>[_1bitnorm] ...
      _[datalesspaz][+][xmlresponse][_<suffix>]

where <suffix> is provided by the user. For example:
"xcorr_1996-2012_xmlresponse"

The files, depending on their extension, contain the following data:

- .pickle       = set of all cross-correlations (instance of
                  pscrosscorr.CrossCorrelationCollection) exported in binary
                  format with module pickle;

- .txt          = all cross-correlations exported in ascii format
                  (one column per pair);

- .stats.txt    = general information on cross-correlations in ascii
                  format: stations coordinates, number of days, inter-
                  station distance etc.

- .stations.txt = general information on the stations: coordinates,
                  nb of cross-correlations in which it appears, total
                  nb of days it has been cross-correlated etc.

- .png          = figure showing all the cross-correlations (normalized to
                  unity), stacked as a function of inter-station distance.
"""

from pysismo import pscrosscorr, pserrors, psstation
import os
import sys
import warnings
import datetime as dt

# ====================================================
# parsing configuration file to import some parameters
# ====================================================

from pysismo.psconfig import (
    MSEED_DIR, DATALESS_DIR, STATIONXML_DIR, CROSSCORR_DIR,
    USE_DATALESSPAZ, USE_STATIONXML, CROSSCORR_STATIONS_SUBSET, CROSSCORR_SKIPLOCS,
    FIRSTDAY, LASTDAY, MINFILL, FREQMIN, FREQMAX, CORNERS, ZEROPHASE, PERIOD_RESAMPLE,
    ONEBIT_NORM, FREQMIN_EARTHQUAKE, FREQMAX_EARTHQUAKE, WINDOW_TIME, WINDOW_FREQ,
    CROSSCORR_TMAX)

print "\nProcessing parameters:"
print "- dir of miniseed data: " + MSEED_DIR
print "- dir of dataless seed data: " + DATALESS_DIR
print "- dir of stationXML data: " + STATIONXML_DIR
print "- output dir: " + CROSSCORR_DIR
print "- band-pass: {:.1f}-{:.1f} s".format(1.0 / FREQMAX, 1.0 / FREQMIN)
if ONEBIT_NORM:
    print "- normalization in time-domain: one-bit normalization"
else:
    s = ("- normalization in time-domain: "
         "running normalization in earthquake band ({:.1f}-{:.1f} s)")
    print s.format(1.0 / FREQMAX_EARTHQUAKE, 1.0 / FREQMIN_EARTHQUAKE)
fmt = '%d/%m/%Y'
s = "- cross-correlation will be stacked between {}-{}"
print s.format(FIRSTDAY.strftime(fmt), LASTDAY.strftime(fmt))
subset = CROSSCORR_STATIONS_SUBSET
if subset:
    print "  for stations: {}".format(', '.join(subset))
print


# ========================================
# Name of output files (without extension).
# E.g., "xcorr_2000-2012_xmlresponse"
# ========================================

responsefrom = []
if USE_DATALESSPAZ:
    responsefrom.append('datalesspaz')
if USE_STATIONXML:
    responsefrom.append('xmlresponse')
OUTBASENAME_PARTS = [
    'xcorr',
    '-'.join(s for s in CROSSCORR_STATIONS_SUBSET) if CROSSCORR_STATIONS_SUBSET else None,
    '{}-{}'.format(FIRSTDAY.year, LASTDAY.year),
    '1bitnorm' if ONEBIT_NORM else None,
    '+'.join(responsefrom)
]
OUTFILESPATH = os.path.join(CROSSCORR_DIR, '_'.join(p for p in OUTBASENAME_PARTS if p))

print 'Default name of output files (without extension):\n"{}"\n'.format(OUTFILESPATH)
suffix = raw_input("Enter suffix to append: [none]\n")
if suffix:
    OUTFILESPATH = u'{}_{}'.format(OUTFILESPATH, suffix)
print 'Results will be exported to files:\n"{}" (+ extension)\n'.format(OUTFILESPATH)

# ============
# Main program
# ============

# Reading inventories in dataless seed and/or StationXML files
dataless_inventories = []
xml_inventories = []
if USE_DATALESSPAZ:
    warnings.filterwarnings('ignore')
    dataless_inventories = psstation.get_dataless_inventories(
        dataless_dir=DATALESS_DIR,
        verbose=True)
    warnings.filterwarnings('default')
    print
if USE_STATIONXML:
    xml_inventories = psstation.get_stationxml_inventories(
        stationxml_dir=STATIONXML_DIR,
        verbose=True)
    print

# Getting list of stations
stations = psstation.get_stations(mseed_dir=MSEED_DIR,
                                  xml_inventories=xml_inventories,
                                  dataless_inventories=dataless_inventories,
                                  startday=FIRSTDAY,
                                  endday=LASTDAY,
                                  verbose=True)

# Initializing collection of cross-correlations
xc = pscrosscorr.CrossCorrelationCollection()

# Loop on day
nday = (LASTDAY - FIRSTDAY).days + 1
dates = [FIRSTDAY + dt.timedelta(days=i) for i in range(nday)]
for date in dates:
    print "\nProcessing data of day ", date

    # Initializing dict of current date's traces, {station name: trace}
    tracedict = {}

    # loop on stations appearing in subdir corresponding to current month
    month_subdir = '{year}-{month:02d}'.format(year=date.year, month=date.month)
    month_stations = sorted(sta for sta in stations if month_subdir in sta.subdirs)

    # subset if stations (if provided)
    if CROSSCORR_STATIONS_SUBSET:
        month_stations = [sta for sta in month_stations
                          if sta.name in CROSSCORR_STATIONS_SUBSET]

    # getting processed trace for each station at current date
    for istation, station in enumerate(month_stations):

        if station == month_stations[-1] and not tracedict:
            # no need to continue with only one station
            s = '[no data: skipping remaining station {}.{}]'
            print s.format(station.network, station.name),
            continue

        # getting processed trace of station at the current date:
        # processing includes band-pass filtering, demeaning,
        # detrending, downsampling, time-normalization and
        # spectral whitening (see function's doc)
        try:
            trace = pscrosscorr.preprocessed_trace(
                station=station,
                date=date,
                dataless_inventories=dataless_inventories,
                xml_inventories=xml_inventories,
                skiplocs=CROSSCORR_SKIPLOCS,
                minfill=MINFILL,
                freqmin=FREQMIN,
                freqmax=FREQMAX,
                freqmin_earthquake=FREQMIN_EARTHQUAKE,
                freqmax_earthquake=FREQMAX_EARTHQUAKE,
                corners=CORNERS,
                zerophase=ZEROPHASE,
                period_resample=PERIOD_RESAMPLE,
                onebit_norm=ONEBIT_NORM,
                window_time=WINDOW_TIME,
                window_freq=WINDOW_FREQ,
                verbose=True)
        except pserrors.CannotPreprocess:
            # cannot preprocess if daily fill < *minfill*, no instrument
            # response was found etc. (see function's doc)
            continue

        # adding processed trace to dict of traces: {station name: trace}
        tracedict[station.name] = trace

    # stacking cross-correlations of the current day
    if len(tracedict) < 2:
        print "\nNo cross-correlation for this day"
        continue

    print '\nStacking cross-correlations'
    xc.add(tracedict=tracedict,
           stations=stations,
           xcorr_tmax=CROSSCORR_TMAX,
           verbose=True)

# exporting cross-correlations
if not xc.pairs():
    print "No cross-correlation could be calculated: nothing to export!"
    sys.exit()

# exporting to binary and ascii files
xc.export(outprefix=OUTFILESPATH, stations=stations, verbose=True)

# exporting to png file
print "Exporting cross-correlations to file: {}.png".format(OUTFILESPATH)
# optimizing time-scale: max time = max distance / vmin (vmin = 2.5 km/s)
maxdist = max([xc[s1][s2].dist() for s1, s2 in xc.pairs()])
maxt = min(CROSSCORR_TMAX, maxdist / 2.5)
xc.plot(xlim=(-maxt, maxt), outfile=OUTFILESPATH + '.png', showplot=False)