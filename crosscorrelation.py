#!/usr/bin/python -u
"""
[Advice: run this script using python with unbuffered output:
`python -u crosscorrelation.py`]

This script reads seismic waveform data from a set of stations, and
calculates the cross-correlations between all pairs of stations
(or optionally displays their amplitude spectra). The data (in
miniseed format) must be located in folder *MSEED_DIR*. The stations
information (coordinates, instrument response) can be read from
dataless seed files (if *USE_DATALESSPAZ* = True) located in
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

from pysismo import pscrosscorr, pserrors, psspectrum, psstation, psutils
import obspy.core
import obspy.core.trace
from obspy.signal import cornFreq2Paz
import obspy.xseed
import numpy as np
from numpy.fft import rfft, irfft
import os
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
    CROSSCORR_TMAX, CALC_SPECTRA, SPECTRA_STATIONS, SPECTRA_FIRSTDAY, SPECTRA_LASTDAY,
    PLOT_TRACES)

print "\nProcessing parameters:"
print "- dir of miniseed data: " + MSEED_DIR
print "- dir of dataless seed data: " + DATALESS_DIR
print "- dir of stationXML data: " + STATIONXML_DIR
print "- output dir: " + CROSSCORR_DIR
print "- band-pass: {:.0f}-{:.0f} s".format(1.0 / FREQMAX, 1.0 / FREQMIN)
if ONEBIT_NORM:
    print "- normalization in time-domain: one-bit normalization"
else:
    s = ("- normalization in time-domain: "
         "running normalization in earthquake band ({:.0f}-{:.0f} s)")
    print s.format(1.0 / FREQMAX_EARTHQUAKE, 1.0 / FREQMIN_EARTHQUAKE)
fmt = '%d/%m/%Y'
if not CALC_SPECTRA:
    s = "- cross-correlation will be stacked between {}-{}"
    print s.format(FIRSTDAY.strftime(fmt), LASTDAY.strftime(fmt))
    subset = CROSSCORR_STATIONS_SUBSET
else:
    s = "- spectra of traces will be estimated between {}-{}"
    print s.format(SPECTRA_FIRSTDAY.strftime(fmt), SPECTRA_LASTDAY.strftime(fmt))
    subset = SPECTRA_STATIONS
if subset:
    print "  for stations: {}".format(', '.join(subset))
print

# ========================
# Constants and parameters
# ========================

EPS = 1.0E-6
ONEDAY = dt.timedelta(days=1)
ONEHOUR = dt.timedelta(hours=1)

# Simulated instrument
PAZ_SIM = cornFreq2Paz(0.01)  # no attenuation up to period 100 s

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

if not CALC_SPECTRA:
    print 'Default name of output files:\n"{}*"\n'.format(OUTFILESPATH)
    suffix = raw_input("Enter suffix to append: [none]\n")
    if suffix:
        OUTFILESPATH = u'{}_{}'.format(OUTFILESPATH, suffix)
    print 'Results will be exported to files:\n"{}*"\n'.format(OUTFILESPATH)

# ============
# Main program
# ============

# Reading inventories in dataless seed and/or StationXML files
datalessinventories = []
xmlinventories = []
if USE_DATALESSPAZ:
    warnings.filterwarnings('ignore')
    datalessinventories = psstation.get_dataless_inventories(
        dataless_dir=DATALESS_DIR,
        verbose=True)
    warnings.filterwarnings('default')
    print
if USE_STATIONXML:
    xmlinventories = psstation.get_stationxml_inventories(
        stationxml_dir=STATIONXML_DIR,
        verbose=True)
    print

# Getting list of stations
stations = psstation.get_stations(
    mseed_dir=MSEED_DIR,
    xmlinventories=xmlinventories,
    datalessinventories=datalessinventories,
    startday=FIRSTDAY if not CALC_SPECTRA else SPECTRA_FIRSTDAY,
    endday=LASTDAY if not CALC_SPECTRA else SPECTRA_LASTDAY,
    verbose=True)

# Initializing collection of cross-correlations
xc = pscrosscorr.CrossCorrelationCollection()
# Initializing spectra list
spectra = psspectrum.SpectrumList()

# Loop on day
nday = LASTDAY - FIRSTDAY if not CALC_SPECTRA else SPECTRA_LASTDAY - SPECTRA_FIRSTDAY
nday = int(nday / (3600.0 * 24.0)) + 1
day1 = FIRSTDAY if not CALC_SPECTRA else SPECTRA_FIRSTDAY
daylist = [day1 + i * ONEDAY for i in range(nday)]
for day in daylist:
    assert isinstance(day, obspy.core.UTCDateTime)  # to avoid warnings in PyCharm
    print "\nProcessing data of day ", day.date

    # Getting and filtering all traces of the day
    # -> tracedict = dict {station: trace}
    tracedict = dict()

    # loop on stations appearing in subdir corresponding to current month
    month_subdir = '{year}-{month:02d}'.format(year=day.year, month=day.month)
    month_stations = sorted(sta for sta in stations if month_subdir in sta.subdirs)

    # subset if stations (if provided)
    if CROSSCORR_STATIONS_SUBSET:
        month_stations = [sta for sta in month_stations
                          if sta.name in CROSSCORR_STATIONS_SUBSET]

    for istation, station in enumerate(month_stations):
        if CALC_SPECTRA and station.name not in SPECTRA_STATIONS:
            continue

        # printing, e.g., | BL.CAUB
        print '{sep}{network}.{name}'.format(sep='| ' if istation else '',
                                             network=station.network,
                                             name=station.name),

        if station == month_stations[-1] and not tracedict and not CALC_SPECTRA:
            print '[no other station: skipped]',
            continue

        # Reading station stream (We add one hour of data to each side
        # in order to avoid edge effects when removing the instrument
        # response. The final trace will be trimmed to one day later.)
        st = obspy.core.read(pathname_or_url=station.getpath(day),
                             starttime=day - ONEHOUR,
                             endtime=day + ONEDAY + ONEHOUR)

        # Removing traces from locations to skip,
        # and traces not from 1st loc if several locs
        psutils.clean_stream(st, skiplocs=CROSSCORR_SKIPLOCS)

        # Data fill for current day (also to verify nb of traces)
        fill = psutils.get_fill(st, starttime=day, endtime=day + ONEDAY)
        if fill < MINFILL:
            print '[{:.0f}% fill: skipped]'.format(fill * 100),
            continue

        # Merging traces, FILLING GAPS WITH LINEAR INTERP
        st.merge(fill_value='interpolate')
        # only one trace should remain
        trace = st[0]
        # to enable auto-completion in PyCharm
        assert isinstance(trace, obspy.core.trace.Trace)

        # =================================
        # Raw trace and instrument response
        # =================================

        # looking for instrument response...
        paz = None
        try:
            # ...first in dataless seed inventories
            paz = psstation.get_paz(channelid=trace.id, t=day,
                                    inventories=datalessinventories)
            print '[paz]',
        except pserrors.NoPAZFound:
            # ... then in dataless seed inventories, replacing 'BHZ' with 'HHZ'
            # in trace's id (trick to make code work with Diogo's data)
            try:
                paz = psstation.get_paz(channelid=trace.id.replace('BHZ', 'HHZ'),
                                        t=day, inventories=datalessinventories)
                print '[paz]',
            except pserrors.NoPAZFound:
                # ...finally in StationXML inventories
                try:
                    trace.attach_response(inventories=xmlinventories)
                    print '[xml]',
                except:
                    print '[no resp: skipped]',
                    continue

        # Stacking power spectrum of station
        if CALC_SPECTRA:
            savetrace = PLOT_TRACES and day == daylist[-1]
            spectra.add(trace=trace, station=station, filters='RAW',
                        starttime=day, endtime=day + ONEDAY, savetrace=savetrace)

        # ============================================
        # Removing instrument response, mean and trend
        # ============================================

        # removing response...
        if paz:
            # ...using paz:
            if trace.stats.sampling_rate > 10.0:
                # decimating large trace, else fft crashes
                factor = int(np.ceil(trace.stats.sampling_rate / 10))
                trace.decimate(factor=factor, no_filter=True)
            trace.simulate(paz_remove=paz, paz_simulate=PAZ_SIM,
                           remove_sensitivity=True, simulate_sensitivity=True,
                           nfft_pow2=True)
        else:
            # ...using StationXML:
            # first band-pass to downsample data before removing response
            # (else remove_response() method is slow or even hangs)
            trace.filter(type="bandpass", freqmin=FREQMIN, freqmax=FREQMAX,
                         corners=CORNERS, zerophase=ZEROPHASE)
            psutils.resample(trace, dt_resample=PERIOD_RESAMPLE)
            trace.remove_response(output="VEL", zero_mean=True)

        # trimming, demeaning, detrending
        trace.trim(starttime=day, endtime=day + ONEDAY)
        trace.detrend(type='constant')
        trace.detrend(type='linear')

        if np.all(trace.data == 0.0):
            # no data -> skipping trace
            print '[only zeros: skipped]',
            continue

        # Stacking power spectrum of station
        if CALC_SPECTRA:
            spectra.add(trace=trace, station=station, filters='RESPONSE',
                        savetrace=savetrace)

        # =========
        # Band-pass
        # =========
        # keeping a copy of the trace to calculate weights of time-normalization
        trcopy = trace.copy()

        # band-pass
        trace.filter(type="bandpass", freqmin=FREQMIN, freqmax=FREQMAX,
                     corners=CORNERS, zerophase=ZEROPHASE)

        # downsampling trace if not already done
        if abs(1.0 / trace.stats.sampling_rate - PERIOD_RESAMPLE) > EPS:
            psutils.resample(trace, dt_resample=PERIOD_RESAMPLE)

        # Stacking power spectrum of station
        if CALC_SPECTRA:
            spectra.add(trace=trace, station=station, filters='BANDPASS',
                        savetrace=savetrace)

        # =====================
        # One-bit normalization
        # =====================
        if ONEBIT_NORM:
            trace.data = np.sign(trace.data)
            # Stacking power spectrum of station
            if CALC_SPECTRA:
                spectra.add(trace=trace, station=station, filters='ONEBITNORM',
                            savetrace=savetrace)

            # skipping all other filters if one-bit normalization
            continue

        # ==================
        # Time-normalization
        # ==================
        # Calculating time-normalization weights (in earthquake band)
        # Applying band-pass in earthquake band
        trcopy.filter(type="bandpass", freqmin=FREQMIN_EARTHQUAKE,
                      freqmax=FREQMAX_EARTHQUAKE, corners=CORNERS,
                      zerophase=ZEROPHASE)
        # decimating trace
        psutils.resample(trcopy, PERIOD_RESAMPLE)

        # Time-normalization weights from smoothed abs(data)
        # Note that trace's data can be a masked array
        halfwindow = int(round(WINDOW_TIME * trcopy.stats.sampling_rate / 2))
        mask = ~trcopy.data.mask if np.ma.isMA(trcopy.data) else None
        tnorm_w = psutils.moving_avg(np.abs(trcopy.data),
                                     halfwindow=halfwindow,
                                     mask=mask)
        if np.ma.isMA(trcopy.data):
            # turning time-normalization weights into a masked array
            print "[Warning: trace's data is a masked array]",
            tnorm_w = np.ma.masked_array(tnorm_w, trcopy.data.mask)

        if np.any((tnorm_w == 0.0) | np.isnan(tnorm_w)):
            # illegal normalizing value -> skipping trace
            print '[zero or NaN normalization weight: skipped]',
            continue

        # time-normalization
        trace.data /= tnorm_w

        # Stacking power spectrum of station
        if CALC_SPECTRA:
            spectra.add(trace=trace, station=station, filters='TIME_NORM',
                        savetrace=savetrace)

        # ==================
        # Spectral whitening
        # ==================
        fft = rfft(trace.data)  # real FFT
        deltaf = trace.stats.sampling_rate / trace.stats.npts  # frequency step
        # smoothing amplitude spectrum
        halfwindow = int(round(WINDOW_FREQ / deltaf / 2.0))
        weight = psutils.moving_avg(abs(fft), halfwindow=halfwindow)
        # normalizing spectrum and back to time domain            
        trace.data = irfft(fft / weight, n=len(trace.data))
        # re bandpass to avoid low/high freq noise
        trace.filter(type="bandpass", freqmin=FREQMIN, freqmax=FREQMAX,
                     corners=CORNERS, zerophase=ZEROPHASE)

        # Stacking power spectrum of station
        if CALC_SPECTRA:
            spectra.add(trace=trace, station=station, filters='SPECTRAL_WHITENING',
                        savetrace=savetrace)

        # ==============================================
        # Verifying that we don't have nan in trace data
        # ==============================================
        if np.any(np.isnan(trace.data)):
            s = u"Got nan at date {date}, in trace:\n{trace}"
            raise Exception(s.format(date=day.date, trace=trace))

        # adding processed trace to dict of traces: {station name: trace}
        tracedict[station.name] = trace

    # ==============================================
    # Stacking cross-correlations of the current day
    # ==============================================
    if not CALC_SPECTRA:
        print '\nStacking cross-correlations'
        xc.add(tracedict=tracedict, stations=stations,
               xcorr_tmax=CROSSCORR_TMAX, verbose=True)

# exporting cross-correlations
if not CALC_SPECTRA:
    # exporting to binary and ascii files
    xc.export(outprefix=OUTFILESPATH, stations=stations, verbose=True)

    # exporting to png file
    print "Exporting cross-correlations to file: {}.png".format(OUTFILESPATH)
    # optimizing time-scale: max time = max distance / vmin (vmin = 2.5 km/s)
    maxdist = max([xc[s1][s2].dist() for s1, s2 in xc.pairs()])
    maxt = min(CROSSCORR_TMAX, maxdist / 2.5)
    xc.plot(xlim=(-maxt, maxt), outfile=OUTFILESPATH + '.png', showplot=False)

# plotting spectra
if CALC_SPECTRA:
    spectra.plot(smooth_window_freq=WINDOW_FREQ)