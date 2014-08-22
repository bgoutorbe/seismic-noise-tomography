#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script reads seismic records from a set of station, and
calculates the cross-correlations between all pairs of stations
(or optionally displays their amplitude spectra).

The procedure consists in stacking daily cross-correlations
between pairs of stations after:
(1) removing the instrument response, the mean and the trend,
(2) band-passing the data,
(3) normalizing the signal with its amplitude in the earthquake
    period band (or one-bit normalizing the signal) and
(4) whitening the amplitude spectrum
"""

from pysismo import pscrosscorr, pserrors, psspectrum, psstation, psutils, psfortran
import obspy.core
import obspy.core.trace
from obspy.signal import cornFreq2Paz
import obspy.xseed
import numpy as np
from numpy.fft import rfft, irfft
import os
import warnings

# ====================================================
# parsing configuration file to import some parameters
# ====================================================

from pysismo.psconfig import (
    CROSSCORR_DIR, USE_DATALESSPAZ, USE_STATIONXML, CROSSCORR_STATIONS_SUBSET,
    FIRSTDAY, LASTDAY, MINFILL, FREQMIN, FREQMAX, CORNERS, ZEROPHASE, PERIOD_RESAMPLE,
    ONEBIT_NORM, FREQMIN_EARTHQUAKE, FREQMAX_EARTHQUAKE, WINDOW_TIME, WINDOW_FREQ,
    CROSSCORR_TMAX, CALC_SPECTRA, SPECTRA_STATIONS, SPECTRA_FIRSTDAY, SPECTRA_LASTDAY,
    PLOT_TRACES)

# ========================
# Constants and parameters
# ========================

EPS = 1.0E-6
ONEDAY = 3600 * 24
EDGE = 3600

# Simulated instrument
PAZ_SIM = cornFreq2Paz(0.01)  # no attenuation up to period 100 s

# ==========
# Out prefix
# ==========

responsefrom = []
if USE_DATALESSPAZ:
    responsefrom.append('datalesspaz')
if USE_STATIONXML:
    responsefrom.append('xmlresponse')
OUTPREFIX_PARTS = [
    'xcorr',
    '-'.join(s for s in CROSSCORR_STATIONS_SUBSET) if CROSSCORR_STATIONS_SUBSET else None,
    '{}-{}'.format(FIRSTDAY.year, LASTDAY.year),
    '1bitnorm' if ONEBIT_NORM else None,
    '+'.join(responsefrom)
]
OUTPREFIX = os.path.join(CROSSCORR_DIR, '_'.join(p for p in OUTPREFIX_PARTS if p))

# ============
# Main program
# ============

if not CALC_SPECTRA:
    print "Cross-correlations will be exported to files {}*\n".format(OUTPREFIX)

# Reading inventories in dataless seed and/or StationXML files
datalessinventories = []
xmlinventories = []
if USE_DATALESSPAZ:
    warnings.filterwarnings('ignore')
    datalessinventories = psstation.get_dataless_inventories(
        dataless_dir=psstation.DATALESS_DIR,
        verbose=True)
    warnings.filterwarnings('default')
    print
if USE_STATIONXML:
    xmlinventories = psstation.get_stationxml_inventories(
        stationxml_dir=psstation.STATIONXML_DIR,
        verbose=True)
    print

# Getting list of stations
stations = psstation.get_stations(
    mseed_dir=psstation.MSEED_DIR,
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
nday = int(nday / ONEDAY) + 1
day1 = FIRSTDAY if not CALC_SPECTRA else SPECTRA_FIRSTDAY
daylist = [day1 + i * ONEDAY for i in range(nday)]
for day in daylist:
    print "\nProcessing data of day ", day.date

    # Getting and filtering all traces of the day
    # -> tracedict = dict {station: trace}
    tracedict = dict()

    # loop on stations appearing in subdir corresponding to current month
    month_subdir = '{year}-{month:02d}'.format(year=day.year, month=day.month)
    month_stations = sorted(sta for sta in stations if month_subdir in sta.subdirs)

    # subset if stations (if provided)
    if CROSSCORR_STATIONS_SUBSET:
        month_stations = [sta for sta in month_stations if sta.name in CROSSCORR_STATIONS_SUBSET]

    for istation, station in enumerate(month_stations):
        assert isinstance(station, psstation.Station)
        if CALC_SPECTRA and station.name not in SPECTRA_STATIONS:
            continue

        # printing, e.g., | BL.CAUB
        print '{sep}{network}.{name}'.format(sep='| ' if istation else '',
                                             network=station.network,
                                             name=station.name),

        if station == month_stations[-1] and not tracedict and not CALC_SPECTRA:
            print '[no other station: skipped]',
            continue

        # Reading station stream
        st = obspy.core.read(pathname_or_url=station.getpath(day),
                             starttime=day - EDGE,
                             endtime=day + ONEDAY + EDGE)

        # Removing traces from locations to skip,
        # and traces not from 1st loc if several locs
        psutils.clean_stream(st, skiplocs=psutils.CROSSCORR_SKIPLOCS)

        # Data fill for current day (also to verify nb of traces)
        fill = psutils.get_fill(st, starttime=day, endtime=day + ONEDAY)
        if fill < MINFILL:
            print '[{:.0f}% fill: skipped]'.format(fill * 100),
            continue

        # Merging traces, FILLING GAPS WITH LINEAR INTERP
        st.merge(fill_value='interpolate')

        # =================================
        # Raw trace and instrument response
        # =================================
        trace = st[0]
        assert isinstance(trace, obspy.core.trace.Trace)

        # looking for instrument response...
        paz = None
        try:
            # ...first in dataless seed inventories
            paz = psstation.get_paz(channelid=trace.id, t=day,
                                    inventories=datalessinventories)
            print '[paz]',
        except pserrors.NoPAZFound:
            # ...then in StationXML inventories
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
        window = round(WINDOW_TIME * trcopy.stats.sampling_rate / 2)
        if not np.ma.isMA(trcopy.data):
            tnorm_w = psfortran.utils.moving_avg(
                np.abs(trcopy.data), window, len(trcopy.data))
        else:
            print '[Warning: masked array]',
            tnorm_w = psfortran.utils.moving_avg_mask(
                np.abs(trcopy.data).data, -trcopy.data.mask,
                window, len(trcopy.data))
            tnorm_w = np.ma.masked_array(data=tnorm_w, mask=trcopy.data.mask)

        if np.any(tnorm_w == 0.0):
            # illegal normalizing value -> skipping trace
            print '[zero norm weight: skipped]',
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
        window = WINDOW_FREQ / deltaf
        weight = psfortran.utils.moving_avg(abs(fft), window)
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

# plotting & writing cross-correlation
if not CALC_SPECTRA:
    s = 'Exporting cross-correlations to files {prefix}.[txt|pickle]'
    print s.format(prefix=OUTPREFIX)
    xc.export(outprefix=OUTPREFIX)
    s = 'Plotting cross-correlations and saving to file {prefix}.png'
    print s.format(prefix=OUTPREFIX)
    xc.plot(xlim=(-1500, 1500), outfile=OUTPREFIX + '.png')

# plotting spectra
if CALC_SPECTRA:
    spectra.plot(smooth_window_freq=WINDOW_FREQ)