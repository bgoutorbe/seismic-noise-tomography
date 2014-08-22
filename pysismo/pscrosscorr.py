#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module managing cross-correlation operations
"""

# todo: find reference dispersion maps
# todo: method group_velocities(periods, whiten=True)

import pserrors, psutils, pstomo, psfortran
import obspy.signal
import obspy.xseed
import obspy.signal.cross_correlation
import obspy.signal.filter
from obspy.core import AttribDict
from obspy.signal.invsim import cosTaper
import numpy as np
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from scipy import integrate
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.optimize import minimize
import itertools as it
import os
import shutil
import glob
import pickle
import copy
from collections import OrderedDict
import datetime as dt
from calendar import monthrange

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec

# ====================================================
# parsing configuration file to import some parameters
# ====================================================
from psconfig import (
    CROSSCORR_DIR, FTAN_DIR, SPECTSNR_BANDS, PLOTXCORR_BANDS,
    RAWFTAN_PERIODS, CLEANFTAN_PERIODS, FTAN_VELOCITIES, FTAN_ALPHA,
    BBOX_LARGE, BBOX_SMALL)

# ========================
# Constants and parameters
# ========================

EPS = 1.0e-5
ONESEC = dt.timedelta(seconds=1)


class MonthYear:
    """
    Hashable class holding a month of a year
    """
    def __init__(self, *args, **kwargs):
        """
        Usage: MonthYear(3, 2012) or MonthYear(month=3, year=2012) or
               MonthYear(date[time](2012, 3, 12))
        """
        if len(args) == 2 and not kwargs:
            month, year = args
        elif not args and set(kwargs.keys()) == {'month', 'year'}:
            month, year = kwargs['month'], kwargs['year']
        elif len(args) == 1 and not kwargs:
            month, year = args[0].month, args[0].year
        else:
            s = ("Usage: MonthYear(3, 2012) or MonthYear(month=3, year=2012) or "
                 "MonthYear(date[time](2012, 3, 12))")
            raise Exception(s)

        self.m = month
        self.y = year

    def __str__(self):
        """
        E.g., 03-2012
        """
        return '{:02d}-{}'.format(self.m, self.y)

    def __repr__(self):
        """
        E.g., <03-2012>
        """
        return '<{}>'.format(str(self))

    def __eq__(self, other):
        """
        Comparison with other, which can be a MonthYear object,
        or a sequence of int (month, year)
        @type other: L{MonthYear} or (int, int)
        """
        try:
            return self.m == other.m and self.y == other.y
        except:
            try:
                return (self.m, self.y) == tuple(other)
            except:
                return False

    def __hash__(self):
        return hash(self.m) ^ hash(self.y)


class MonthCrossCorrelation:
    """
    Class holding cross-correlation over a single month
    """
    def __init__(self, month, ndata):
        """
        @type month: L{MonthYear}
        @type ndata: int
        """
        # attaching month and year
        self.month = month

        # initializing stats
        self.nday = 0

        # data array of month cross-correlation
        self.dataarray = np.zeros(ndata)

    def monthfill(self):
        """
        Returns the relative month fill (between 0-1)
        """
        return float(self.nday) / monthrange(year=self.month.y, month=self.month.m)[1]

    def __repr__(self):
        s = '<cross-correlation over single month {}: {} days>'
        return s.format(self.month, self.nday)


class CrossCorrelation:
    """
    Cross-correlation class, which contains:
    - a pair of stations
    - a pair of sets of locations (from trace.location)
    - a pair of sets of ids (from trace.id)
    - start day, end day and nb of days of cross-correlation
    - distance between stations
    - a time array and a (cross-correlation) data array
    """

    def __init__(self, station1, station2, xcorr_dt=1, xcorr_tmax=2000):
        """
        @type station1: L{pysismo.psstation.Station}
        @type station2: L{pysismo.psstation.Station}
        @type xcorr_dt: float
        @type xcorr_tmax: float
        """
        # pair of stations
        self.station1 = station1
        self.station2 = station2

        # locations and trace ids of stations
        self.locs1 = set()
        self.locs2 = set()
        self.ids1 = set()
        self.ids2 = set()

        # initializing stats
        self.startday = None
        self.endday = None
        self.nday = 0

        # initializing time and data arrays of cross-correlation
        nmax = int(xcorr_tmax / xcorr_dt)
        self.timearray = np.arange(-nmax * xcorr_dt, (nmax + 1)*xcorr_dt, xcorr_dt)
        self.dataarray = np.zeros(2 * nmax + 1)

        #  has cross-corr been symmetrized? whitened?
        self.symmetrized = False
        self.whitened = False

        # initializing list of cross-correlations over a single month
        self.monthxcs = []

    def __repr__(self):
        s = '<cross-correlation between stations {0}-{1}: avg {2} days>'
        return s.format(self.station1.name, self.station2.name, self.nday)

    def __str__(self):
        """
        E.g., 'Cross-correlation between stations SPB['10'] - ITAB['00','10']:
               365 days from 2002-01-01 to 2002-12-01'
        """
        locs1 = ','.join(sorted("'{}'".format(loc) for loc in self.locs1))
        locs2 = ','.join(sorted("'{}'".format(loc) for loc in self.locs2))
        s = ('Cross-correlation between stations '
             '{sta1}[{locs1}]-{sta2}[{locs2}]: '
             '{nday} days from {start} to {end}')
        return s.format(sta1=self.station1.name, locs1=locs1,
                        sta2=self.station2.name, locs2=locs2, nday=self.nday,
                        start=self.startday, end=self.endday)

    def dist(self):
        """
        Geodesic distance (in km) between stations, using the
        WGS-84 ellipsoidal model of the Earth
        """
        return self.station1.dist(self.station2)

    def copy(self):
        """
        Makes a copy of self
        """
        # shallow copy
        result = copy.copy(self)
        # copy of month cross-correlations
        result.monthxcs = [copy.copy(mxc) for mxc in self.monthxcs]
        return result

    def add(self, tr1, tr2):
        """
        Stacks cross-correlation between 2 traces
        @type tr1: L{obspy.core.trace.Trace}
        @type tr2: L{obspy.core.trace.Trace}
        """
        # verifying sampling rates
        try:
            assert 1.0 / tr1.stats.sampling_rate == self._get_xcorr_dt()
            assert 1.0 / tr2.stats.sampling_rate == self._get_xcorr_dt()
        except AssertionError:
            s = 'Sampling rates of traces are not equal:\n{tr1}\n{tr2}'
            raise Exception(s.format(tr1=tr1, tr2=tr2))

        # cross-correlation
        xcorr = obspy.signal.cross_correlation.xcorr(
            tr1, tr2, shift_len=self._get_xcorr_nmax(), full_xcorr=True)[2]
        # verifying that we don't have NaN
        if np.any(np.isnan(xcorr)):
            s = u"Got NaN in cross-correlation between traces:\n{tr1}\n{tr2}"
            raise pserrors.NaNError(s.format(tr1=tr1, tr2=tr2))

        # stacking cross-corr
        self.dataarray += xcorr
        # updating stats: 1st day, last day, nb of days of cross-corr
        startday = (tr1.stats.starttime + ONESEC).date
        self.startday = min(self.startday, startday) if self.startday else startday
        endday = (tr1.stats.endtime - ONESEC).date
        self.endday = max(self.endday, endday) if self.endday else endday
        self.nday += 1

        # stacking cross-corr over single month
        month = MonthYear((tr1.stats.starttime + ONESEC).date)
        try:
            monthxc = next(monthxc for monthxc in self.monthxcs
                           if monthxc.month == month)
        except StopIteration:
            # appending new month xc
            monthxc = MonthCrossCorrelation(month=month, ndata=len(self.timearray))
            self.monthxcs.append(monthxc)
        monthxc.dataarray += xcorr
        monthxc.nday += 1

        # updating (adding) locs and ids
        self.locs1.add(tr1.stats.location)
        self.locs2.add(tr2.stats.location)
        self.ids1.add(tr1.id)
        self.ids2.add(tr2.id)

    def symmetrize(self, inplace=False):
        """
        Symmetric component of cross-correlation (including
        the list of cross-corr over a single month).
        Returns self if already symmetrized or inPlace=True

        @rtype: CrossCorrelation
        """

        if self.symmetrized:
            # already symmetrized
            return self

        # symmetrizing on self or copy of self
        xcout = self if inplace else self.copy()

        n = len(xcout.timearray)
        mid = (n - 1) / 2

        # verifying that time array is symmetric wrt 0
        if n % 2 != 1:
            raise Exception('Cross-correlation cannot be symmetrized')
        if not np.alltrue(xcout.timearray[mid:] + xcout.timearray[mid::-1] < EPS):
            raise Exception('Cross-correlation cannot be symmetrized')

        # calculating symmetric component of cross-correlation
        xcout.timearray = xcout.timearray[mid:]
        for obj in [xcout] + (xcout.monthxcs if hasattr(xcout, 'monthxcs') else []):
            a = obj.dataarray
            obj.dataarray = (a[mid:] + a[mid::-1]) / 2.0

        xcout.symmetrized = True
        return xcout

    def whiten(self, inplace=False, window_freq=0.002,
               bandpass_tmin=7.0, bandpass_tmax=150):
        """
        Spectral whitening of cross-correlation (including
        the list of cross-corr over a single month).
        @rtype: CrossCorrelation
        """
        if hasattr(self, 'whitened') and self.whitened:
            # already whitened
            return self

        # whitening on self or copy of self
        xcout = self if inplace else self.copy()

        # frequency step
        npts = len(xcout.timearray)
        sampling_rate = 1.0 / xcout._get_xcorr_dt()
        deltaf = sampling_rate / npts

        # loop over cross-corr and one-month stacks
        for obj in [xcout] + (xcout.monthxcs if hasattr(xcout, 'monthxcs') else []):
            a = obj.dataarray
            # Fourier transform
            ffta = rfft(a)

            # smoothing amplitude spectrum
            window = int(window_freq / deltaf)
            weight = psfortran.utils.moving_avg(abs(ffta), window)
            a[:] = irfft(ffta / weight, n=npts)

            # bandpass to avoid low/high freq noise
            obj.dataarray = psutils.bandpass(data=a, df=1.0 / xcout._get_xcorr_dt(),
                                             tmin=bandpass_tmin, tmax=bandpass_tmax)

        xcout.whitened = True
        return xcout

    def SNR(self, bands=None, whiten=False, vmin=2.5, vmax=5.0,
            noise_window_trail=500, noise_window=500, months=None):
        """
        [spectral] signal-to-noise ratio
        @type bands: (list of (float, float))
        @type whiten: bool
        @type vmin: float
        @type vmax: bool
        @type noise_window_trail: float
        @type noise_window: float
        @type months: list of (L{MonthYear} or (int, int))
        @rtype: L{numpy.ndarray}
        """
        # symmetric part of cross-corr
        xcout = self.symmetrize(inplace=False)

        # spectral whitening
        if whiten:
            xcout = xcout.whiten(inplace=False)

        # cross-corr of desired months
        xcdata = xcout._get_monthyears_xcdataarray(months=months)

        SNR = []
        if not bands:
            bands = [None]
        for band in bands:
            if band:
                dataarray = psutils.bandpass(data=xcdata,
                                             df=1.0 / xcout._get_xcorr_dt(),
                                             tmin=band[0], tmax=band[1])
            else:
                dataarray = xcdata

            # signal window
            tau_min = xcout.dist() / vmax
            tau_max = xcout.dist() / vmin
            window = (xcout.timearray <= tau_max) & (xcout.timearray >= tau_min)
            peak = np.max(abs(dataarray[window]))

            # noise window
            tau_min = tau_max + noise_window_trail
            tau_max = tau_min + noise_window
            window = (xcout.timearray <= tau_max) & (xcout.timearray >= tau_min)
            noise = dataarray[window].std()

            # appending SNR
            SNR.append(peak / noise)

        # returning 1d array if spectral SNR, 0d array if normal SNR
        return np.array(SNR) if len(SNR) > 1 else np.array(SNR[0])

    def plot(self, whiten=False, sym=False, vmin=2.5, vmax=5.0, months=None):
        """
        Plots cross-correlation and its spectrum
        """
        xcout = self.symmetrize(inplace=False) if sym else self
        if whiten:
            xcout = xcout.whiten(inplace=False)

        # cross-corr of desired months
        xcdata = xcout._get_monthyears_xcdataarray(months=months)

        # cross-correlation plot ===
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(xcout.timearray, xcdata)
        plt.xlabel('Time (s)')
        plt.ylabel('Cross-correlation')
        plt.grid()

        # vmin, vmax
        vkwargs = {
            'fontsize': 8,
            'horizontalalignment': 'center',
            'bbox': dict(color='k', facecolor='white')}
        if vmin:
            ylim = plt.ylim()
            plt.plot(2 * [xcout.dist() / vmin], ylim, color='grey')
            xy = (xcout.dist() / vmin, plt.ylim()[0])
            plt.annotate('{0} km/s'.format(vmin), xy=xy, xytext=xy, **vkwargs)
            plt.ylim(ylim)

        if vmax:
            ylim = plt.ylim()
            plt.plot(2 * [xcout.dist() / vmax], ylim, color='grey')
            xy = (xcout.dist() / vmax, plt.ylim()[0])
            plt.annotate('{0} km/s'.format(vmax), xy=xy, xytext=xy, **vkwargs)
            plt.ylim(ylim)

        # title
        plt.title(xcout._plottitle(months=months))

        # spectrum plot ===
        plt.subplot(2, 1, 2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.grid()

        # frequency and amplitude arrays
        npts = len(xcdata)
        nfreq = npts / 2 + 1 if npts % 2 == 0 else (npts + 1) / 2
        sampling_rate = 1.0 / xcout._get_xcorr_dt()
        freqarray = np.arange(nfreq) * sampling_rate / npts
        amplarray = np.abs(rfft(xcdata))
        plt.plot(freqarray, amplarray)
        plt.xlim((0.0, 0.2))

        plt.show()

    def plot_by_period_band(self, axlist=None, plot_title=True, whiten=False,
                            vmin=2.5, vmax=5.0, months=None, outfile=None):
        """
        Plots cross-correlation for various bands of periods
        (PLOTXCORR_BANDS)

        @type axlist: list of L{matplotlib.axes.AxesSubplot}
        """
        # one plot per band + plot of original xcorr
        nplot = len(PLOTXCORR_BANDS) + 1

        # limits of time axis
        xlim = (0, min(1.5 * self.dist() / vmin, self.timearray.max()))

        # creating figure if not given as input
        fig = None
        if not axlist:
            fig = plt.figure()
            axlist = [fig.add_subplot(nplot, 1, i) for i in range(1, nplot + 1)]

        axlist[0].get_figure().subplots_adjust(hspace=0)

        # symmetrization
        xcout = self.symmetrize(inplace=False)

        # spectral whitening
        if whiten:
            xcout = xcout.whiten(inplace=False)

        # cross-corr of desired months
        xcdata = xcout._get_monthyears_xcdataarray(months=months)

        # plotting original cross-correlation
        axlist[0].plot(xcout.timearray, xcdata)
        xy = (axlist[0].get_xlim()[0] + 10, axlist[0].get_ylim()[1] / 2)
        axlist[0].annotate('Original data', xy=xy, xytext=xy,
                           bbox={'color': 'k', 'facecolor': 'white'})
        # xlims
        _ = axlist[0].set_xlim(xlim)
        axlist[0].grid(True)
        # removing labels
        axlist[0].set_xticklabels([])
        axlist[0].set_yticklabels([])

        # title
        if plot_title:
            title = xcout._plottitle(prefix='Cross-corr. ', months=months)
            axlist[0].set_title(title)

        # vmin, vmax
        vkwargs = {
            'fontsize': 8,
            'horizontalalignment': 'center',
            'bbox': dict(color='k', facecolor='white')}
        if vmin:
            ylim = axlist[0].get_ylim()
            axlist[0].plot(2 * [xcout.dist() / vmin], ylim, color='grey')
            xy = (xcout.dist() / vmin, ylim[0] + 0.1 * (ylim[1] - ylim[0]))
            axlist[0].annotate(s='{0} km/s'.format(vmin), xy=xy, xytext=xy, **vkwargs)
            axlist[0].set_ylim(ylim)

        if vmax:
            ylim = axlist[0].get_ylim()
            axlist[0].plot(2 * [xcout.dist() / vmax], ylim, color='grey')
            xy = (xcout.dist() / vmax, ylim[0] + 0.1 * (ylim[1] - ylim[0]))
            axlist[0].annotate(s='{0} km/s'.format(vmax), xy=xy, xytext=xy, **vkwargs)
            axlist[0].set_ylim(ylim)

        # plotting band-filtered cross-correlation
        for ax, (tmin, tmax) in zip(axlist[1:], PLOTXCORR_BANDS):
            lastplot = ax is axlist[-1]

            dataarray = psutils.bandpass(data=xcdata, df=1.0 / xcout._get_xcorr_dt(),
                                         tmin=tmin, tmax=tmax)
            ax.plot(xcout.timearray, dataarray)
            xy = (ax.get_xlim()[0] + 10, ax.get_ylim()[1] / 2)
            ax.annotate(s='{tmin} - {tmax} s'.format(tmin=tmin, tmax=tmax),
                        xy=xy, xytext=xy, bbox={'color': 'k', 'facecolor': 'white'})
            ax.grid(True)

            # vmin, vmax
            if vmin:
                ylim = ax.get_ylim()
                ax.plot(2 * [xcout.dist() / vmin], ylim, color='grey')
                ax.set_ylim(ylim)
            if vmax:
                ylim = ax.get_ylim()
                ax.plot(2 * [xcout.dist() / vmax], ylim, color='grey')
                ax.set_ylim(ylim)
            # removing labels
            if not lastplot:
                ax.set_xticklabels([])
            ax.set_yticklabels([])
            # xlims
            ax.set_xlim(xlim)
            # axis title
            if lastplot:
                ax.set_xlabel('Time (s)')

        if outfile:
            axlist[0].gcf().savefig(outfile, dpi=300, transparent=True)

        if fig:
            fig.show()

    def FTAN(self, whiten=True, phase_corr=None, months=None, vgarray_init=None):
        """
        Frequency-time analysis of a cross-correlation signal.
        Calculates the Fourier transform of the cross-correlation,
        calculates the analytic signal in frequency domain,
        applies Gaussian bandpass filters centered around given
        center periods, and calculates the filtered analytic
        signal back in time domain.

        Returns the amplitude matrix A(T0,v) and phase matrix phi(T0,v),
        that is, the amplitude and phase function of velocity v of the
        analytic signal filtered around period T0.
        Also extracts and returns the group velocity curve from the
        amplitude matrix (an initial guess my be given to accelerate
        vg curve extraction).

        FTAN periods in variable *FTAN_PERIODS*
        FTAN velocities in variable *FTAN_VELOCITIES*

        @type whiten: bool
        @type phase_corr: L{scipy.interpolate.interpolate.interp1d}
        @type months: list of (L{MonthYear} or (int, int))
        @type vgarray_init: L{numpy.ndarray}
        @rtype: (L{numpy.ndarray}, L{numpy.ndarray}, L{DispersionCurve})
        """
        # no phase correction given <=> raw FTAN
        raw_ftan = phase_corr is None
        ftan_periods = RAWFTAN_PERIODS if raw_ftan else CLEANFTAN_PERIODS

        # getting the symmetrized cross-correlation
        xcout = self.symmetrize(inplace=False)
        # whitening cross-correlation
        if whiten:
            xcout = xcout.whiten(inplace=False)

        # cross-corr of desired months
        xcdata = xcout._get_monthyears_xcdataarray(months=months)
        if xcdata is None:
            raise Exception('No data to perform FTAN in selected months')

        # FTAN analysis: amplitute and phase function of
        # center periods T0 and time t
        ampl, phase = FTAN(x=xcdata,
                           dt=xcout._get_xcorr_dt(),
                           periods=ftan_periods,
                           alpha=FTAN_ALPHA,
                           phase_corr=phase_corr)

        # re-interpolating amplitude and phase as functions
        # of center periods T0 and velocities v
        tne0 = xcout.timearray != 0.0
        x = ftan_periods                                 # x = periods
        y = (self.dist() / xcout.timearray[tne0])[::-1]  # y = velocities
        zampl = ampl[:, tne0][:, ::-1]                   # z = amplitudes
        zphase = phase[:, tne0][:, ::-1]                 # z = phases
        # spline interpolation
        ampl_interp_func = RectBivariateSpline(x, y, zampl)
        phase_interp_func = RectBivariateSpline(x, y, zphase)
        # re-sampling over periods and velocities
        ampl_resampled = ampl_interp_func(ftan_periods, FTAN_VELOCITIES)
        phase_resampled = phase_interp_func(ftan_periods, FTAN_VELOCITIES)

        # extracting the group velocity curve from the amplitude matrix,
        # that is, the velocity curve that maximizes amplitude and best
        # avoids jumps, in the interval of clean FTAN periods
        periodmask = ((np.array(ftan_periods) >= min(CLEANFTAN_PERIODS)) &
                      (np.array(ftan_periods) <= max(CLEANFTAN_PERIODS)))
        vgarray = _extract_vgarray(amplmatrix=ampl_resampled,
                                   velocities=FTAN_VELOCITIES,
                                   periodmask=periodmask,
                                   optimizecurve=raw_ftan,
                                   vgarray_init=vgarray_init)

        vgcurve = pstomo.DispersionCurve(periods=ftan_periods,
                                         v=vgarray,
                                         station1=self.station1,
                                         station2=self.station2)

        return ampl_resampled, phase_resampled, vgcurve

    def FTAN_complete(self, whiten=True, months=None, add_SNRs=True):
        """
        Frequency-time analysis including phase-matched filter and
        seasonal variability:

        (1) Performs a FTAN of the raw cross-correlation signal,
        (2) Uses the raw group velocities to calculate the phase corr.
        (3) Performs a FTAN with the phase correction
            ("phase matched filter")
        (4) Repeats the procedure for all 12 trimesters if no
            list of months is given

        Optionally, adds spectral SNRs at the periods of the clean
        vg curve.

        Returns raw ampl, raw vg, cleaned ampl, cleaned vg.

        @type whiten: bool
        @type months: list of (L{MonthYear} or (int, int))
        @type add_SNRs: bool
        @rtype: (L{numpy.ndarray}, L{numpy.ndarray},
                 L{numpy.ndarray}, L{DispersionCurve})
        """
        # symmetrized, whitened cross-corr
        xc = self.symmetrize(inplace=False)
        if whiten:
            xc = xc.whiten(inplace=False)

        # raw FTAN (no need to whiten any more)
        rawampl, _, rawvg = xc.FTAN(whiten=False, months=months)

        # phase function from raw vg curve
        phase_corr = xc.phase_func(vgcurve=rawvg,
                                   whiten=False,
                                   months=months)

        # clean FTAN
        cleanampl, _, cleanvg = xc.FTAN(whiten=False,
                                        phase_corr=phase_corr,
                                        months=months)

        # adding spectral SNRs associated with the periods of the
        # clean vg curve
        if add_SNRs:
            cleanvg.add_SNRs(xc, months=months)

        if months is None:
            # set of available months (without year)
            available_months = set(mxc.month.m for mxc in xc.monthxcs)

            # extracting clean vg curves for all 12 trimesters:
            # Jan-Feb-March, Feb-March-Apr ... Dec-Jan-Feb
            for trimester_start in range(1, 13):
                # months of trimester, e.g. [1, 2, 3], [2, 3, 4] ... [12, 1, 2]
                trimester_months = [(trimester_start + i - 1) % 12 + 1
                                    for i in range(3)]
                # do we have data in all months?
                if any(month not in available_months for month in trimester_months):
                    continue
                # list of month-year whose month belong to current trimester
                months_of_xc = [mxc.month for mxc in xc.monthxcs
                                if mxc.month.m in trimester_months]

                # raw-clean FTAN on trimester data, using the vg curve
                # extracted from all data as initial guess
                _, _, rawvg_trimester = xc.FTAN(whiten=False,
                                                months=months_of_xc,
                                                vgarray_init=rawvg.v)

                phase_corr_trimester = xc.phase_func(vgcurve=rawvg_trimester,
                                                     whiten=False,
                                                     months=months_of_xc)

                _, _, cleanvg_trimester = xc.FTAN(whiten=False,
                                                  phase_corr=phase_corr_trimester,
                                                  months=months_of_xc,
                                                  vgarray_init=cleanvg.v)

                # adding spectral SNRs associated with the periods of the
                # clean trimester vg curve
                if add_SNRs:
                    cleanvg_trimester.add_SNRs(xc, months=months_of_xc)

                # adding trimester vg curve
                cleanvg.add_trimester(trimester_start, cleanvg_trimester)

        return rawampl, rawvg, cleanampl, cleanvg

    def phase_func(self, vgcurve=None, whiten=True, months=None):
        """
        Calculates the phase from the group velocity obtained
        using method self.FTAN, following the relationship:

        k(f) = 2.pi.integral[ 1/vg(f'), f'=f0..f ]
        phase(f) = distance.k(f)

        Returns the function phase: freq -> phase(freq)

        @param vgcurve: group velocity curve
        @type vgcurve: L{DispersionCurve}
        @type whiten: bool
        @type months: list of (L{MonthYear} or (int, int))
        @rtype: L{scipy.interpolate.interpolate.interp1d}
        """
        # FTAN analysis to extract array of group velocities
        # if not provided by user
        if vgcurve is None:
            vgcurve = self.FTAN(whiten=whiten, months=months)[2]

        freqarray = 1.0 / vgcurve.periods[::-1]
        vgarray = vgcurve.v[::-1]

        # array k[f]
        k = np.zeros_like(freqarray)
        k[0] = 0.0
        k[1:] = 2 * np.pi * integrate.cumtrapz(y=1.0 / vgarray, x=freqarray)

        # array phi[f]
        phi = k * self.dist()

        # phase function of f
        return interp1d(x=freqarray, y=phi)

    def plot_FTAN(self, rawampl=None, rawvg=None, cleanampl=None, cleanvg=None,
                  whiten=True, showplot=True, months=None, bbox=BBOX_SMALL,
                  figsize=(16, 5), outfile=None):
        """
        Plots log[ampl(T0,v)²], where ampl(T0,v) is the amplitude
        from the FTAN analysis, and the group velocity curve,
        before (raw FTAN) and after (clean FTAN) the phase matched filter.

        Returns the plot figure.

        @type rawampl: L{numpy.ndarray}
        @type rawvg: L{DispersionCurve}
        @type cleanampl: L{numpy.ndarray}
        @type cleanvg: L{DispersionCurve}
        @type showplot: bool
        @type whiten: bool
        @type months: list of (L{MonthYear} or (int, int))
        @rtype: L{matplotlib.figure.Figure}
        """
        # performing FTAN analysis if needed
        if any(obj is None for obj in [rawampl, rawvg, cleanampl, cleanvg]):
            rawampl, rawvg, cleanampl, cleanvg = self.FTAN_complete(whiten=whiten,
                                                                    months=months,
                                                                    add_SNRs=True)

        # preparing figure
        fig = plt.figure(figsize=figsize)

        # =======================================================
        # 1th panel: cross-correlation (original and band-passed)
        # =======================================================

        gs1 = gridspec.GridSpec(len(PLOTXCORR_BANDS) + 1, 1, wspace=0.0, hspace=0.0)
        axlist = [fig.add_subplot(ss) for ss in gs1]
        self.plot_by_period_band(axlist=axlist, plot_title=False, whiten=whiten)

        # ===================
        # 2st panel: raw FTAN
        # ===================

        gs2 = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.0)
        ax = fig.add_subplot(gs2[0, 0])

        extent = (min(RAWFTAN_PERIODS), max(RAWFTAN_PERIODS),
                  min(FTAN_VELOCITIES), max(FTAN_VELOCITIES))
        m = np.log10(rawampl.transpose() ** 2)
        ax.imshow(m, aspect='auto', origin='lower', extent=extent)
        ax.set_xlabel("period (sec)")
        ax.set_ylabel("Velocity (km/sec)")
        # saving limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # raw & clean vg curves
        ax.plot(rawvg.periods, rawvg.v, color='blue', linestyle='dashed',
                lw=2, label='raw FTAN')
        ax.plot(cleanvg.periods, cleanvg.v, color='black',
                lw=2, label='clean FTAN')
        ax.legend()
        # plotting cut-off period
        cutoffperiod = self.dist() / 12.0
        ax.plot([cutoffperiod, cutoffperiod], ylim, color='grey')
        # setting initial extent
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # =====================
        # 3nd panel: clean FTAN
        # =====================
        ax = fig.add_subplot(gs2[0, 1])

        extent = (min(CLEANFTAN_PERIODS), max(CLEANFTAN_PERIODS),
                  min(FTAN_VELOCITIES), max(FTAN_VELOCITIES))
        m = np.log10(cleanampl.transpose() ** 2)
        ax.imshow(m, aspect='auto', origin='lower', extent=extent)
        ax.set_xlabel("period (sec)")
        ax.set_ylabel("Velocity (km/sec)")
        # saving limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # trimester vg curves
        ntrimester = len(cleanvg.v_trimesters)
        for i, vg_trimester in enumerate(cleanvg.filtered_trimester_vels()):
            label = '{} trimester FTANs'.format(ntrimester) if i == 0 else None
            ax.plot(cleanvg.periods, vg_trimester, color='gray', label=label)

        # clean vg curve + error bars
        vels, sdevs = cleanvg.filtered_vels_sdevs()
        ax.errorbar(x=cleanvg.periods, y=vels, yerr=sdevs, color='black',
                    lw=2, label='clean FTAN')
        ax.legend()
        # plotting cut-off period
        cutoffperiod = self.dist() / 12.0
        ax.plot([cutoffperiod, cutoffperiod], ylim, color='grey')
        # setting initial extent
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # ======================================
        # 4rd panel: tectonic provinces and pair
        # ======================================

        gs3 = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.0)
        ax = fig.add_subplot(gs3[0, 0])

        psutils.basemap(ax, labels=False, axeslabels=False)
        x = (self.station1.coord[0], self.station2.coord[0])
        y = (self.station1.coord[1], self.station2.coord[1])
        s = (self.station1.name, self.station2.name)
        ax.plot(x, y, '^-', color='k', ms=10, mfc='w', mew=1)
        for lon, lat, label in zip(x, y, s):
            ax.text(lon, lat, label, ha='center', va='bottom', fontsize=7, weight='bold')
        ax.set_xlim(bbox[:2])
        ax.set_ylim(bbox[2:])

        # adjusting sizes
        gs1.update(left=0.02, right=0.25)
        gs2.update(left=0.30, right=0.83)
        gs3.update(left=0.85, right=0.98)

        # figure title
        title = self._FTANplot_title(whiten=whiten, months=months)
        fig.suptitle(title, fontsize=14)

        # exporting to file
        if outfile:
            fig.savefig(outfile, dpi=300, transparent=True)

        if showplot:
            plt.show()
        return fig

    def _plottitle(self, prefix='', months=None):
        """
        E.g., 'SPB-ITAB (365 days from 2002-01-01 to 2002-12-01)'
           or 'SPB-ITAB (90 days in months 01-2002, 02-2002)'
        """
        s = '{pref}{sta1}-{sta2} '
        s = s.format(pref=prefix, sta1=self.station1.name, sta2=self.station2.name)
        if not months:
            nday = self.nday
            s += '({} days from {} to {})'.format(
                nday, self.startday.strftime('%d/%m/%Y'),
                self.endday.strftime('%d/%m/%Y'))
        else:
            monthxcs = [mxc for mxc in self.monthxcs if mxc.month in months]
            nday = sum(monthxc.nday for monthxc in monthxcs)
            strmonths = ', '.join(str(m.month) for m in monthxcs)
            s += '{} days in months {}'.format(nday, strmonths)
        return s

    def _FTANplot_title(self, whiten=True, months=None):
        """
        E.g., 'BL.GNSB-IU.RCBR, dist=1781 km, SNR=28.8, min spect SNR=24.1, ndays=208'
        """
        minSNR = min(self.SNR(bands=SPECTSNR_BANDS, whiten=whiten, months=months))
        if not months:
            nday = self.nday
        else:
            nday = sum(monthxc.nday for monthxc in self.monthxcs
                       if monthxc.month in months)
        title = u"{}-{}, dist={:.0f} km, SNR={:.1f}, min spect SNR={:.1f}, ndays={}"
        title = title.format(self.station1.network + '.' + self.station1.name,
                             self.station2.network + '.' + self.station2.name,
                             self.dist(), float(self.SNR()), minSNR, nday)
        return title

    def _get_xcorr_dt(self):
        """
        Returns the interval of the time array.
        Warning: no check is made to ensure that that interval is constant.
        @rtype: float
        """
        return self.timearray[1] - self.timearray[0]

    def _get_xcorr_nmax(self):
        """
        Returns the max index of time array:
        - self.timearray = [-t[nmax] ... t[0] ... t[nmax]] if not symmetrized
        -                = [t[0] ... t[nmax-1] t[nmax]] if symmetrized
        @rtype: int
        """
        nt = len(self.timearray)
        return (nt - 1) / 2 if not self.symmetrized else nt - 1

    def _get_monthyears_xcdataarray(self, months=None):
        """
        Returns the sum of cross-corr data arrays of given
        list of (month,year) -- or the whole cross-corr if
        monthyears is None.

        @type months: list of (L{MonthYear} or (int, int))
        @rtype: L{numpy.ndarray}
        """
        if not months:
            return self.dataarray
        else:
            monthxcs = [mxc for mxc in self.monthxcs if mxc.month in months]
            if monthxcs:
                return sum(monthxc.dataarray for monthxc in monthxcs)
            else:
                return None


class CrossCorrelationCollection(AttribDict):
    """
    Collection of cross-correlations
    = AttribDict{station1.name: AttribDict {station2.name: instance of CrossCorrelation}}

    Usage: self[s1][s2]... or self.s1.s2...
    """

    def __init__(self):
        """
        Initializing object as AttribDict
        """
        AttribDict.__init__(self)

    def __repr__(self):
        npair = len(self.pairs())
        s = '(AttribDict)<Collection of cross-correlation between {0} pairs>'
        return s.format(npair)

    def pairs(self, sort=False, minday=1, minSNR=None, mindist=None,
              withnets=None, onlywithnets=None, pairs_subset=None):
        """
        Returns pairs of stations of cross-correlation collection
        verifying conditions

        @type sort: bool
        @type minday: int
        @type minSNR: float
        @type mindist: float
        @type withnets: list of str
        @type onlywithnets: list of str
        @type pairs_subset: list of (str, str)
        @rtype: list of (str, str)
        """
        pairs = [(s1, s2) for s1 in self for s2 in self[s1]]
        if sort:
            pairs.sort()

        # filtering subset of pairs
        if pairs_subset:
            pairs_subset = [set(pair) for pair in pairs_subset]
            pairs = [pair for pair in pairs if set(pair) in pairs_subset]

        # filtering by nb of days
        pairs = [(s1, s2) for (s1, s2) in pairs
                 if self[s1][s2].nday >= minday]

        # filtering by min SNR
        if minSNR:
            pairs = [(s1, s2) for (s1, s2) in pairs if self[s1][s2].SNR() >= minSNR]

        # filtering by distance
        if mindist:
            pairs = [(s1, s2) for (s1, s2) in pairs
                     if self[s1][s2].dist() >= mindist]

        # filtering by network
        if withnets:
            # one of the station of the pair must belong to networks
            pairs = [(s1, s2) for (s1, s2) in pairs if
                     self[s1][s2].station1.network in withnets or
                     self[s1][s2].station2.network in withnets]
        if onlywithnets:
            # both stations of the pair must belong to networks
            pairs = [(s1, s2) for (s1, s2) in pairs if
                     self[s1][s2].station1.network in onlywithnets and
                     self[s1][s2].station2.network in onlywithnets]

        return pairs

    def pairs_and_SNRarrays(self, pairs_subset=None, minspectSNR=None,
                            whiten=False, verbose=False):
        """
        Returns pairs and spectral SNR array whose spectral SNRs
        are all >= minspectSNR

        Returns {pair1: SNRarray1, pair2: SNRarray2 etc.}

        @type pairs_subset: list of (str, str)
        @type minspectSNR: float
        @type whiten: bool
        @type verbose: bool
        @rtype: dict from (str, str) to L{numpy.ndarray}
        """

        if verbose:
            print "Estimating spectral SNR of pair:",

        # initial list of pairs
        pairs = pairs_subset if pairs_subset else self.pairs()

        # filetring by min spectral SNR
        SNRarraydict = {}
        for (s1, s2) in pairs:
            if verbose:
                print '{0}-{1}'.format(s1, s2),

            SNRarray = self[s1][s2].SNR(bands=SPECTSNR_BANDS, whiten=whiten)
            if not minspectSNR or min(SNRarray) >= minspectSNR:
                SNRarraydict[(s1, s2)] = SNRarray

        if verbose:
            print

        return SNRarraydict

    def add(self, tracedict, stations, xcorr_tmax, verbose=False):
        """
        Stacks cross-correlation from a dict of {station.name: Trace}.

        Initializes self[station1][station2] as an instance of CrossCorrelation
        if the pair station1-station2 is not in self

        @type tracedict: dict from str to L{obspy.core.trace.Trace}
        @type stations: list of L{pysismo.psstation.Station}
        @type xcorr_tmax: float
        @type verbose: bool
        """

        stationtrace_pairs = it.combinations(sorted(tracedict.items()), 2)
        for (s1name, tr1), (s2name, tr2) in stationtrace_pairs:
            if verbose:
                print "{s1}-{s2}".format(s1=s1name, s2=s2name),

            # checking that sampling rates are equal
            assert tr1.stats.sampling_rate == tr2.stats.sampling_rate

            # looking for s1 and s2 in the list of stations
            station1 = next(s for s in stations if s.name == s1name)
            station2 = next(s for s in stations if s.name == s2name)

            # initializing self[s1] if s1 not in self
            # (avoiding setdefault() since behavior in unknown with AttribDict)
            if s1name not in self:
                self[s1name] = AttribDict()

            # initializing self[s1][s2] if s2 not in self[s1]
            if s2name not in self[s1name]:
                self[s1name][s2name] = CrossCorrelation(
                    station1=station1,
                    station2=station2,
                    xcorr_dt=1.0 / tr1.stats.sampling_rate,
                    xcorr_tmax=xcorr_tmax)

            # stacking cross-correlation
            try:
                self[s1name][s2name].add(tr1, tr2)
            except pserrors.NaNError:
                # got NaN
                s = "Warning: got NaN in cross-corr between {s1}-{s2} -> skipping"
                print s.format(s1=s1name, s2=s2name)

        if verbose:
            print

    def plot(self, plot_type='distance', norm=True, whiten=False, sym=False,
             minSNR=None, minday=1, withnets=None, onlywithnets=None,
             outfile=None, xlim=None, figsize=(21.0, 12.0), dpi=300):
        """
        method to plot a collection of cross-correlations
        """

        # preparing pairs
        pairs = self.pairs(minday=minday, minSNR=minSNR, withnets=withnets,
                           onlywithnets=onlywithnets)
        npair = len(pairs)
        if not npair:
            print "Nothing to plot!"
            return

        plt.figure()

        # classic plot = one plot for each pair
        if plot_type == 'classic':
            nrow = int(np.sqrt(npair))
            if np.sqrt(npair) != nrow:
                nrow += 1

            ncol = int(npair / nrow)
            if npair % nrow != 0:
                ncol += 1

            # sorting pairs alphabetically
            pairs.sort()

            for iplot, (s1, s2) in enumerate(pairs):
                # symmetrizing cross-corr if necessary
                xcplot = self[s1][s2].symmetrize(inplace=False) if sym else self[s1][s2]

                # spectral whitening
                if whiten:
                    xcplot = xcplot.whiten(inplace=False)

                # subplot
                plt.subplot(nrow, ncol, iplot + 1)

                # normalizing factor
                nrm = max(abs(xcplot.dataarray)) if norm else 1.0

                # plotting
                plt.plot(xcplot.timearray, xcplot.dataarray / nrm, 'r')
                if xlim:
                    plt.xlim(xlim)

                # title
                locs1 = ','.join(sorted(["'{0}'".format(loc) for loc in xcplot.locs1]))
                locs2 = ','.join(sorted(["'{0}'".format(loc) for loc in xcplot.locs2]))
                s = '{s1}[{locs1}]-{s2}[{locs2}]: {nday} days from {t1} to {t2}'
                title = s.format(s1=s1, locs1=locs1, s2=s2, locs2=locs2,
                                 nday=xcplot.nday, t1=xcplot.startday,
                                 t2=xcplot.endday)
                plt.title(title)

                # x-axis label
                if iplot + 1 == npair:
                    plt.xlabel('Time (s)')

        # distance plot = one plot for all pairs, y-shifted according to pair distance
        elif plot_type == 'distance':
            maxdist = max(self[x][y].dist() for (x, y) in pairs)
            corr2km = maxdist / 30.0
            cc = mpl.rcParams['axes.color_cycle']  # color cycle

            # sorting pairs by distance
            pairs.sort(key=lambda (s1, s2): self[s1][s2].dist())
            for ipair, (s1, s2) in enumerate(pairs):
                # symmetrizing cross-corr if necessary
                xcplot = self[s1][s2].symmetrize(inplace=False) if sym else self[s1][s2]

                # spectral whitening
                if whiten:
                    xcplot = xcplot.whiten(inplace=False)

                # color
                color = cc[ipair % len(cc)]

                # normalizing factor
                nrm = max(abs(xcplot.dataarray)) if norm else 1.0

                # plotting
                xarray = xcplot.timearray
                yarray = xcplot.dist() + corr2km * xcplot.dataarray / nrm
                plt.plot(xarray, yarray, color=color)
                if xlim:
                    plt.xlim(xlim)

                # adding annotation @ xytest, annotation line @ xyarrow
                xmin, xmax = plt.xlim()
                xextent = plt.xlim()[1] - plt.xlim()[0]
                ymin = -0.1 * maxdist
                ymax = 1.1 * maxdist
                if npair <= 40:
                    # all annotations on the right side
                    x = xmax - xextent / 10.0
                    y = maxdist if npair == 1 else ymin + ipair*(ymax-ymin)/(npair-1)
                    xytext = (x, y)
                    xyarrow = (x - xextent / 30.0, xcplot.dist())
                    align = 'left'
                    relpos = (0, 0.5)
                else:
                    # alternating right/left
                    sign = 2 * (ipair % 2 - 0.5)
                    x = xmin + xextent / 10.0 if sign > 0 else xmax - xextent / 10.0
                    y = ymin + ipair / 2 * (ymax - ymin) / (npair / 2 - 1.0)
                    xytext = (x, y)
                    xyarrow = (x + sign * xextent / 30.0, xcplot.dist())
                    align = 'right' if sign > 0 else 'left'
                    relpos = (1, 0.5) if sign > 0 else (0, 0.5)
                net1 = xcplot.station1.network
                net2 = xcplot.station2.network
                locs1 = ','.join(sorted(["'{0}'".format(loc) for loc in xcplot.locs1]))
                locs2 = ','.join(sorted(["'{0}'".format(loc) for loc in xcplot.locs2]))
                s = '{net1}.{s1}[{locs1}]-{net2}.{s2}[{locs2}]: {nday} days {t1}-{t2}'
                s = s.format(net1=net1, s1=s1, locs1=locs1, net2=net2, s2=s2,
                             locs2=locs2, nday=xcplot.nday,
                             t1=xcplot.startday.strftime('%d/%m/%y'),
                             t2=xcplot.endday.strftime('%d/%m/%y'))

                bbox = {'color': color, 'facecolor': 'white', 'alpha': 0.9}
                arrowprops = {'arrowstyle': "-", 'relpos': relpos, 'color': color}

                plt.annotate(s=s, xy=xyarrow, xytext=xytext, fontsize=9,
                             color='k', horizontalalignment=align,
                             bbox=bbox, arrowprops=arrowprops)

            plt.grid()
            plt.xlabel('Time (s)')
            plt.ylabel('Distance (km)')
            plt.ylim((0, plt.ylim()[1]))

        # saving figure
        if outfile:
            if os.path.exists(outfile):
                # backup
                shutil.copyfile(outfile, outfile + '~')
            fig = plt.gcf()
            fig.set_size_inches(figsize)
            fig.savefig(outfile, dpi=dpi)

        # showing plot
        plt.show()

    def plot_spectral_SNR(self, whiten=False, minSNR=None, minspectSNR=None,
                          minday=1, mindist=None, withnets=None, onlywithnets=None):
        """
        Plots spectral SNRs
        """

        # filtering pairs
        pairs = self.pairs(minday=minday, minSNR=minSNR, mindist=mindist,
                           withnets=withnets, onlywithnets=onlywithnets)

        # SNRarrays = dict {(station1,station2): SNR array}
        SNRarrays = self.pairs_and_SNRarrays(pairs_subset=pairs,
                                             minspectSNR=minspectSNR,
                                             whiten=whiten,
                                             verbose=True)

        npair = len(SNRarrays)
        if not npair:
            print 'Nothing to plot!!!'
            return

        # min-max SNR
        minSNR = min([SNR for SNRarray in SNRarrays.values() for SNR in SNRarray])
        maxSNR = max([SNR for SNRarray in SNRarrays.values() for SNR in SNRarray])

        # sorting SNR arrays by increasing first value
        SNRarrays = OrderedDict(sorted(SNRarrays.items(), key=lambda (k, v): v[0]))

        # array of mid of time bands
        periodarray = [(tmin + tmax) / 2.0 for (tmin, tmax) in SPECTSNR_BANDS]
        minperiod = min(periodarray)

        # color cycle
        cc = mpl.rcParams['axes.color_cycle']

        # plotting SNR arrays
        plt.figure()
        for ipair, ((s1, s2), SNRarray) in enumerate(SNRarrays.items()):
            xc = self[s1][s2]
            color = cc[ipair % len(cc)]

            # SNR vs period
            plt.plot(periodarray, SNRarray, color=color)

            # annotation
            xtext = minperiod - 4
            ytext = minSNR * 0.5 + ipair * (maxSNR - minSNR * 0.5) / (npair - 1)
            xytext = (xtext, ytext)
            xyarrow = (minperiod - 1, SNRarray[0])
            relpos = (1, 0.5)
            net1 = xc.station1.network
            net2 = xc.station2.network

            s = '{i}: {net1}.{s1}-{net2}.{s2}: {dist:.1f} km, {nday} days'
            s = s.format(i=ipair, net1=net1, s1=s1, net2=net2, s2=s2,
                         dist=xc.dist(), nday=xc.nday)

            bbox = {'color': color, 'facecolor': 'white', 'alpha': 0.9}
            arrowprops = {'arrowstyle': '-', 'relpos': relpos, 'color': color}

            plt.annotate(s=s, xy=xyarrow, xytext=xytext, fontsize=9,
                         color='k', horizontalalignment='right',
                         bbox=bbox, arrowprops=arrowprops)

        plt.xlim((0.0, plt.xlim()[1]))
        plt.xlabel('Period (s)')
        plt.ylabel('SNR')
        plt.title(u'{0} pairs'.format(npair))
        plt.grid()
        plt.show()

    def plot_pairs(self, minSNR=None, minspectSNR=None, minday=1, mindist=None,
                   withnets=None, onlywithnets=None, pairs_subset=None, whiten=False,
                   stationlabel=False, bbox=BBOX_LARGE, xsize=10, **plotkwargs):
        """
        Plots pairs of stations on a map
        @type bbox: tuple
        """

        # filtering pairs
        pairs = self.pairs(minday=minday, minSNR=minSNR, mindist=mindist,
                           withnets=withnets, onlywithnets=onlywithnets,
                           pairs_subset=pairs_subset)

        if minspectSNR:
            # plotting only pairs with all spect SNR >= minspectSNR
            SNRarraydict = self.pairs_and_SNRarrays(
                pairs_subset=pairs,
                minspectSNR=minspectSNR,
                whiten=whiten,
                verbose=True)
            pairs = SNRarraydict.keys()

        # nb of pairs
        npair = len(pairs)
        if not npair:
            print 'Nothing to plot!!!'
            return

        # initializing figure
        aspectratio = (bbox[3] - bbox[2]) / (bbox[1] - bbox[0])
        plt.figure(figsize=(xsize, aspectratio * xsize))

        # plotting coasts and tectonic provinces
        psutils.basemap(plt.gca(), bbox=bbox)

        # plotting pairs
        for s1, s2 in pairs:
            x, y = zip(self[s1][s2].station1.coord, self[s1][s2].station2.coord)
            if not plotkwargs:
                plotkwargs = dict(color='grey', lw=0.5)
            plt.plot(x, y, '-', **plotkwargs)

        # plotting stations
        x, y = zip(*[s.coord for s in self.stations(pairs)])
        plt.plot(x, y, '^', color='k', ms=10, mfc='w', mew=1)
        if stationlabel:
            # stations label
            for station in self.stations(pairs):
                plt.text(station.coord[0], station.coord[1], station.name,
                         ha='center', va='bottom', fontsize=10, weight='bold')

        # setting axes
        plt.title(u'{0} pairs'.format(npair))
        plt.xlim(bbox[:2])
        plt.ylim(bbox[2:])
        plt.show()

    def export(self, outprefix):
        """
        Exports cross-correlations to picke file and txt file

        @type outprefix: str or unicode
        """
        self._to_picklefile(outprefix)
        self._write_crosscorrs(outprefix)
        self._write_pairsstats(outprefix)
        self._write_stations(outprefix)

    def FTANs(self, prefix=None, suffix='', whiten=True, mindist=None,
              minSNR=None, minspectSNR=None, monthyears=None):
        """
        Export raw-clean FTAN plots to pdf (one page per pair)
        and clean dispersion curves to pickle file, in dir
        *FTAN_DIR*

        @type prefix: str or unicode
        @type suffix: str or unicode
        @type minSNR: float
        @type mindist: float
        @type minspectSNR: float
        @type whiten: bool
        @type monthyears: list of (int, int)
        """
        # setting default file name if not given
        if not prefix:
            parts = ['FTAN']
            if whiten:
                parts.append('whitenedxc')
            if mindist:
                parts.append('mindist={0}'.format(mindist))
            if minSNR:
                parts.append('minSNR={0}'.format(minSNR))
            if minspectSNR:
                parts.append('minspectSNR={0}'.format(minspectSNR))
            if monthyears:
                parts.extend('{:02d}-{}'.format(m, y) for m, y in monthyears)
            else:
                startyear = min(self[s1][s2].startday.year for s1, s2 in self.pairs())
                endyear = max(self[s1][s2].endday.year for s1, s2 in self.pairs())
                parts.append('{}-{}'.format(startyear, endyear))
        else:
            parts = [prefix]
        if suffix:
            parts.append(suffix)
        basename = os.path.join(FTAN_DIR, '_'.join(parts))

        # opening pdf file
        pdf = PdfPages(basename + '.pdf')

        # filtering pairs
        pairs = self.pairs(sort=True, minSNR=minSNR, mindist=mindist)
        if minspectSNR:
            # plotting only pairs with all spect SNR >= minspectSNR
            SNRarraydict = self.pairs_and_SNRarrays(pairs_subset=pairs,
                                                    minspectSNR=minspectSNR,
                                                    whiten=whiten,
                                                    verbose=True)
            pairs = sorted(SNRarraydict.keys())

        print "Plotting FTANs of {0} pairs".format(len(pairs))

        cleanvgcurves = []
        print "Appending FTAN of pair:",
        for i, (s1, s2) in enumerate(pairs):
            # appending FTAN plot of pair s1-s2 to pdf
            print "[{0}] {1}-{2}".format(i + 1, s1, s2),
            xc = self[s1][s2]
            assert isinstance(xc, CrossCorrelation)

            # complete FTAN analysis
            rawampl, rawvg, cleanampl, cleanvg = xc.FTAN_complete(whiten=whiten,
                                                                  months=monthyears)

            # plotting raw-clean FTAN
            fig = xc.plot_FTAN(rawampl, rawvg, cleanampl, cleanvg, showplot=False)
            pdf.savefig(fig)
            plt.close()

            # appending clean vg curve
            cleanvgcurves.append(cleanvg)

        # closing pdf
        pdf.close()

        # exporting vg curves to pickle file
        print '\nExporting clean vg disperion curves to file {}.pickle'.format(basename)
        f = psutils.openandbackup(basename + '.pickle', mode='wb')
        pickle.dump(cleanvgcurves, f, protocol=2)
        f.close()

    def stations(self, pairs, sort=True):
        """
        Returns a list of unique stations corresponding
        to a list of pairs (of station name).

        @type pairs: list of (str, str)
        @rtype: list of L{pysismo.psstation.Station}
        """
        stations = []
        for s1, s2 in pairs:
            if self[s1][s2].station1 not in stations:
                stations.append(self[s1][s2].station1)
            if self[s1][s2].station2 not in stations:
                stations.append(self[s1][s2].station2)

        if sort:
            stations.sort(key=lambda obj: obj.name)

        return stations

    def _to_picklefile(self, outprefix):
        """
        Dumps cross-correlations to (binary) pickle file

        @type outprefix: str or unicode
        """
        f = psutils.openandbackup(outprefix + '.pickle', mode='wb')
        pickle.dump(self, f, protocol=2)
        f.close()

    def _write_crosscorrs(self, outprefix):
        """
        Exports cross-correlations to txt file

        @type outprefix: str or unicode
        """

        # writing data file: time array (1st column)
        # and cross-corr array (one column per pair)
        f = psutils.openandbackup(outprefix + '.txt', mode='w')
        pairs = [(s1, s2) for (s1, s2) in self.pairs(sort=True) if self[s1][s2].nday]

        # writing header
        header = ['time'] + ["{0}-{1}".format(s1, s2) for s1, s2 in pairs]
        f.write('\t'.join(header) + '\n')

        # writing line = ith [time, cross-corr 1st pair, cross-corr 2nd pair etc]
        data = zip(self._get_timearray(), *[self[s1][s2].dataarray for s1, s2 in pairs])
        for fields in data:
            line = [str(fld) for fld in fields]
            f.write('\t'.join(line) + '\n')
        f.close()

    def _write_pairsstats(self, outprefix):
        """
        Exports pairs statistics to txt file

        @type outprefix: str or unicode
        """
        # writing stats file: coord, locations, ids etc. for each pair
        pairs = self.pairs(sort=True)
        f = psutils.openandbackup(outprefix + '.stats.txt', mode='w')
        # header
        header = ['pair', 'lon1', 'lat1', 'lon2', 'lat2',
                  'locs1', 'locs2', 'ids1', 'ids2',
                  'distance', 'startday', 'endday', 'nday']
        f.write('\t'.join(header) + '\n')

        # fields
        for (s1, s2) in pairs:
            fields = [
                '{0}-{1}'.format(s1, s2),
                self[s1][s2].station1.coord[0],
                self[s1][s2].station1.coord[1],
                self[s1][s2].station2.coord[0],
                self[s1][s2].station2.coord[1],
                ','.join(sorted("'{}'".format(l) for l in self[s1][s2].locs1)),
                ','.join(sorted("'{}'".format(l) for l in self[s1][s2].locs2)),
                ','.join(sorted(sid for sid in self[s1][s2].ids1)),
                ','.join(sorted(sid for sid in self[s1][s2].ids2)),
                self[s1][s2].dist(),
                self[s1][s2].startday,
                self[s1][s2].endday,
                self[s1][s2].nday
            ]
            line = [str(fld) if (fld or fld == 0) else 'none' for fld in fields]
            f.write('\t'.join(line) + '\n')

        f.close()

    def _write_stations(self, outprefix):
        """
        Exports information on cross-correlated stations
        to txt file

        @type outprefix: str or unicode
        """
        # stations with at least one day of cross-correlation
        # with another stations
        stations = self.stations(self.pairs(minday=1), sort=True)

        # opening stations file and writing:
        # station name, network, lon, lat, nb of pairs, total days of cross-corr
        f = psutils.openandbackup(outprefix + '.stations.txt', mode='w')
        header = ['name', 'network', 'lon', 'lat', 'npair', 'nday']
        f.write('\t'.join(header) + '\n')

        for station in stations:
            # pairs in which station appears
            pairs = [(s1, s2) for s1, s2 in self.pairs()
                     if station in [self[s1][s2].station1, self[s1][s2].station2]]
            # total nb of days of pairs
            nday = sum(self[s1][s2].nday for s1, s2 in pairs)
            # writing fields
            fields = [
                station.name,
                station.network,
                str(station.coord[0]),
                str(station.coord[1]),
                str(len(pairs)),
                str(nday)
            ]
            f.write('\t'.join(fields) + '\n')

        f.close()

    def _get_timearray(self):
        """
        Returns time array of cross-correlations

        @rtype: L{numpy.ndarray}
        """

        pairs = self.pairs()

        # reference time array
        s1, s2 = pairs[0]
        reftimearray = self[s1][s2].timearray

        # checking that all time arrays are equal to reference time array
        for (s1, s2) in pairs:
            if np.any(self[s1][s2].timearray != reftimearray):
                s = 'Cross-corr collection does not have a unique timelag array'
                raise Exception(s)

        return reftimearray


def load_pickled_xcorr(pickle_file):
    """
    Loads pickle-dumped cross-correlations

    @type pickle_file: str or unicode
    @rtype: L{CrossCorrelationCollection}
    """
    f = open(name=pickle_file, mode='rb')
    xc = pickle.load(f)
    f.close()
    return xc


def load_pickled_xcorr_interactive(xcorr_dir=CROSSCORR_DIR, xcorr_files='xcorr*.pickle*'):
    """
    Loads interactively pickle-dumped cross-correlations, by giving the user
    a choice among a list of file matching xcorrFiles

    @type xcorr_dir: str or unicode
    @type xcorr_files: str or unicode
    @rtype: L{CrossCorrelationCollection}
    """

    # looking for files that match xcorrFiles
    pathxcorr = os.path.join(xcorr_dir, xcorr_files)
    flist = glob.glob(pathname=pathxcorr)
    flist.sort()

    pickle_file = None
    if len(flist) == 1:
        pickle_file = flist[0]
        print 'Reading cross-correlation from file ' + pickle_file
    elif len(flist) > 0:
        print 'Select file containing cross-correlations:'
        print '\n'.join('{i} - {f}'.format(i=i, f=os.path.basename(f))
                        for (i, f) in enumerate(flist))
        i = int(raw_input('\n'))
        pickle_file = flist[i]

    # loading cross-correlations
    xc = load_pickled_xcorr(pickle_file=pickle_file)

    return xc


def FTAN(x, dt, periods, alpha, phase_corr=None):
    """
    Frequency-time analysis of a time series.
    Calculates the Fourier transform of the signal (xarray),
    calculates the analytic signal in frequency domain,
    applies Gaussian bandpass filters centered around given
    center periods, and calculates the filtered analytic
    signal back in time domain.
    Returns the amplitude/phase matrices A(f0,t) and phi(f0,t),
    that is, the amplitude/phase function of time t of the
    analytic signal filtered around period T0 = 1 / f0.

    @param dt: sample spacing
    @type dt: float
    @param x: data array
    @type x: L{numpy.ndarray}
    @param periods: center periods around of Gaussian bandpass filters
    @type periods: L{numpy.ndarray} or list
    @param alpha: smoothing parameter of Gaussian filter
    @type alpha: float
    @param phase_corr: phase correction, function of freq
    @type phase_corr: L{scipy.interpolate.interpolate.interp1d}
    @rtype: (L{numpy.ndarray}, L{numpy.ndarray})
    """

    # Initializing amplitude/phase matrix: each column =
    # amplitude function of time for a given Gaussian filter
    # centered around a period
    amplitude = np.zeros(shape=(len(periods), len(x)))
    phase = np.zeros(shape=(len(periods), len(x)))

    # Fourier transform
    Xa = fft(x)
    # aray of frequencies
    freq = fftfreq(len(Xa), d=dt)

    # analytic signal in frequency domain:
    #         | 2X(f)  for f > 0
    # Xa(f) = | X(f)   for f = 0
    #         | 0      for f < 0
    # with X = fft(x)
    Xa[freq < 0] = 0.0
    Xa[freq > 0] *= 2.0

    # applying phase correction: replacing phase with given
    # phase function of freq
    if phase_corr:
        # doamin of definition of phase_corr(f)
        minfreq = phase_corr.x.min()
        maxfreq = phase_corr.x.max()
        mask = (freq >= minfreq) & (freq <= maxfreq)

        # replacing phase with user-provided phase correction:
        # updating Xa(f) as |Xa(f)|.exp(-i.phase_corr(f))
        phi = phase_corr(freq[mask])
        Xa[mask] = np.abs(Xa[mask]) * np.exp(-1j * phi)

        # tapering
        taper = cosTaper(npts=mask.sum(), p=0.05)
        Xa[mask] *= taper
        Xa[-mask] = 0.0

    # applying narrow bandpass Gaussian filters
    for iperiod, T0 in enumerate(periods):
        # bandpassed analytic signal
        f0 = 1.0 / T0
        Xa_f0 = Xa * np.exp(-alpha * ((freq - f0) / f0) ** 2)
        # back to time domain
        xa_f0 = ifft(Xa_f0)
        # filling amplitude and phase of column
        amplitude[iperiod, :] = np.abs(xa_f0)
        phase[iperiod, :] = np.angle(xa_f0)

    return amplitude, phase


def _extract_vgarray(amplmatrix, velocities, periodmask=None, optimizecurve=True,
                     vgarray_init=None):
    """
    Extracts a group velocity vg vs period T from an amplitude
    matrix *ampl*, itself obtained from FTAN.

    Among the curves that ride along local maxima of amplitude,
    the selected group velocity curve vg(T) maximizes the sum of
    amplitudes, while preserving some smoothness (minimizing of
    *_vg_minimizes*), over periods covered by *periodmask*.
    The curve can be furthered optimized using a minimization
    algorithm.

    If an initial vg array is given (*vgarray_init*), then only
    the optimization algorithm is applied, using *vgarray_init*
    as starting point.

    ampl[i, j] = amplitude at period nb i and velocity nb j

    @type amplmatrix: L{numpy.ndarray}
    @type velocities: L{numpy.ndarray}
    @type vgarray_init: L{numpy.ndarray}
    @rtype: L{numpy.ndarray}
    """
    # if an initial guess for vg array is given, we simply apply
    # the optimization procedure using it as starting guess
    if not vgarray_init is None:
        return _optimize_vg(amplmatrix=amplmatrix,
                            velocities=velocities,
                            vg0=vgarray_init)[0]

    nperiods = amplmatrix.shape[0]

    # building list of possible (vg, ampl) curves at all periods
    vgampl_arrays = None
    for iperiod in range(nperiods):
        # local maxima of amplitude at period nb *iperiod*
        argsmax = psutils.local_maxima_indices(amplmatrix[iperiod, :])

        if not argsmax:
            # no local minimum => leave nan in (vg, ampl) curves
            continue

        if not vgampl_arrays:
            # initialzing the list of possible (vg, ampl) curves with local maxima
            # at current period, and nan elsewhere
            vgampl_arrays = [(np.zeros(nperiods) * np.nan, np.zeros(nperiods) * np.nan)
                             for _ in range(len(argsmax))]
            for argmax, (vgarray, amplarray) in zip(argsmax, vgampl_arrays):
                vgarray[iperiod] = velocities[argmax]
                amplarray[iperiod] = amplmatrix[iperiod, argmax]
            continue

        # inserting the velocities that locally maximizes amplitude
        # to the correct curves
        for argmax in argsmax:
            # velocity that locally maximizes amplitude (potential group vel)
            vg = velocities[argmax]

            # we select the (vg, ampl) curve for which the jump wrt previous
            # vg (not nan) is minimum
            lastvg = lambda vgarray: vgarray[:iperiod][-np.isnan(vgarray[:iperiod])][-1]
            vgjump = lambda (vgarray, amplarray): abs(lastvg(vgarray) - vg)
            vgarray, amplarray = min(vgampl_arrays, key=vgjump)

            # if the curve already has a vg attributed at this period, we
            # duplicate it
            if not np.isnan(vgarray[iperiod]):
                vgarray, amplarray = copy.copy(vgarray), copy.copy(amplarray)
                vgampl_arrays.append((vgarray, amplarray))

            # inserting (vg, ampl) at current period to the selected curve
            vgarray[iperiod] = vg
            amplarray[iperiod] = amplmatrix[iperiod, argmax]

        # filling curves without (vg, ampl) data at the current period
        unfilledcurves = [(vgarray, amplarray) for vgarray, amplarray in vgampl_arrays
                          if np.isnan(vgarray[iperiod])]
        for vgarray, amplarray in unfilledcurves:
            # inserting vel (which locally maximizes amplitude) for which
            # the jump wrt the previous (not nan) vg of the curve is minimum
            lastvg = vgarray[:iperiod][-np.isnan(vgarray[:iperiod])][-1]
            vgjump = lambda arg: abs(lastvg - velocities[arg])
            argmax = min(argsmax, key=vgjump)
            vgarray[iperiod] = velocities[argmax]
            amplarray[iperiod] = amplmatrix[iperiod, argmax]

    # amongst possible vg curves, we select the one that maximizes amplitude,
    # while preserving some smoothness
    def funcmin((vgarray, amplarray)):
        if not periodmask is None:
            return _vg_minimizes(vgarray[periodmask], amplarray[periodmask])
        else:
            return _vg_minimizes(vgarray, amplarray)
    vgarray, _ = min(vgampl_arrays, key=funcmin)

    # filling holes of vg curve
    masknan = np.isnan(vgarray)
    if masknan.any():
        vgarray[masknan] = np.interp(x=masknan.nonzero()[0],
                                     xp=(-masknan).nonzero()[0],
                                     fp=vgarray[-masknan])

    # further optimizing curve using a minimization algorithm
    if optimizecurve:
        # first trying with initial guess = the one above
        vgcurve1, funcmin1 = _optimize_vg(amplmatrix=amplmatrix,
                                          velocities=velocities,
                                          vg0=vgarray)
        # then trying with initial guess = constant velocity 3 km/s
        vgcurve2, funcmin2 = _optimize_vg(amplmatrix=amplmatrix,
                                          velocities=velocities,
                                          vg0=3.0*np.ones(nperiods))
        vgarray = vgcurve1 if funcmin1 <= funcmin2 else vgcurve2

    return vgarray


def _optimize_vg(amplmatrix, velocities, vg0):
    """
    Optimizing vg curve, i.e., looking for curve that
    minimizes *_vg_minimizes*.

    Returns optimized vg curve and the corresponding
    value of the objective function to minimize

    @type amplmatrix: L{numpy.ndarray}
    @type velocities: L{numpy.ndarray}
    @rtype: L{numpy.ndarray}, float
    """
    nperiods = amplmatrix.shape[0]

    # function that returns the amplitude curve
    # a given input vg curve goes through
    ixperiods = np.arange(nperiods)
    amplcurvefunc2d = RectBivariateSpline(ixperiods, velocities, amplmatrix, kx=1, ky=1)
    amplcurvefunc = lambda vgcurve: amplcurvefunc2d.ev(ixperiods, vgcurve)

    def funcmin(vgcurve, verbose=False):
        """Objective function to minimize"""
        # amplitude curve corresponding to vg vurve
        return _vg_minimizes(vgcurve, amplcurvefunc(vgcurve), verbose=verbose)

    bounds = nperiods * [(min(velocities) + 0.1, max(velocities) - 0.1)]
    method = 'SLSQP'  # methods with bounds: L-BFGS-B, TNC, SLSQP
    resmin = minimize(fun=funcmin, x0=vg0, method=method, bounds=bounds)
    vgcurve = resmin['x']
    #_ = funcmin(vgcurve, verbose=True)

    return vgcurve, resmin['fun']


def _vg_minimizes(vgarray, amplarray, penalizezigzags=False, verbose=False):
    """
    Objective function that vg curve must minimize.
    @type vgarray: L{numpy.ndarray}
    @type amplarray: L{numpy.ndarray}
    """
    # removing nans
    notnan = -(np.isnan(vgarray) | np.isnan(amplarray))
    vgarray = vgarray[notnan]
    amplarray = amplarray[notnan]

    # jumps
    dvg = vgarray[1:] - vgarray[:-1]
    sumdvg2 = sum(dvg**2)

    # zigzags
    sumzigzags = 0.0
    if penalizezigzags:
        iextrema = (psutils.local_maxima_indices(vgarray, include_edges=False) +
                    psutils.local_maxima_indices(-vgarray, include_edges=False))
        if len(iextrema) >= 2:
            iextrema = np.array(sorted(set(iextrema)))
            diextrema = iextrema[1:] - iextrema[:-1]
            dvgextrema = vgarray[iextrema][1:] - vgarray[iextrema][:-1]
            # small-scale zigzags are penalized
            sumzigzags = sum(np.abs(dvgextrema / diextrema))

    # amplitude
    sumamplitude = amplarray.sum()

    if verbose:
        msg = 'sum dvg2: {}\nsum zigzags: {}\nsum ampl: {}'
        print msg.format(sumdvg2, sumzigzags, sumamplitude)

    # vg curve must maximize amplitude and minimize jumps
    return -sumamplitude + sumdvg2 + sumzigzags


if __name__ == '__main__':
    # loading pickled cross-correlations
    xc = load_pickled_xcorr_interactive()
    print "Cross-correlations available in variable 'xc':"
    print xc