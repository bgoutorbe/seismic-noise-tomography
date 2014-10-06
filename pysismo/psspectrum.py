"""
Module related to Fourier spectrum of trace
"""


import psutils
import numpy as np
from numpy.fft import rfft
import matplotlib.pyplot as plt
import itertools as it


class FreqAmplSpectrum:
    """
    Frequence-amplitude spectrum
    """

    def __init__(self, trace):
        """
        @type trace: L{obspy.core.trace.Trace}
        """

        # initializing frequency and amplitude arrays
        npts = trace.stats.npts
        nfreq = npts / 2 + 1 if npts % 2 == 0 else (npts + 1) / 2

        self.freq = np.arange(nfreq) * trace.stats.sampling_rate / npts
        self.coef = np.zeros(nfreq, dtype=np.complex64)

        # adding spectrum of trace
        self.add(trace)

    def __str__(self):
        """
        e.g., Fourier spectrum in frequency interval 0-150 Hz
        @rtype: str
        """
        s = "Fourier spectrum in frequency interval {minf:.1f}-{maxf:.1f} Hz"
        return s.format(minf=min(self.freq), maxf=max(self.freq))

    def __repr__(self):
        return "<{0}>".format(self.__str__())

    def add(self, trace):
        """
        Adds (spectrum of) trace to spectrum
        @type trace: L{obspy.core.trace.Trace}
        """
        self.coef += rfft(trace.data)


class SpectrumInfos:
    """
    Infos on freq-ampl spectrum:
    station, filters, saved trace, freq-ampl spectrum
    """

    def __init__(self, station, filters, trace, savetrace=False):
        """
        @type station: L{Station}
        @type filters: str
        @type trace: L{obspy.core.trace.Trace}
        """
        self.station = station
        self.filters = filters
        self.savedtrace = trace if savetrace else None
        self.spectrum = FreqAmplSpectrum(trace=trace)

    def __repr__(self):
        """
        e.g., <Fourier spectrum in frequency interval 0-150 Hz, station BL.10.NUPB,
               filters 'RAW'>
        @rtype: str
        """
        s = "<{strspect}, station {net}.{ch}.{name}, filters '{filters}'>"
        return s.format(strspect=self.spectrum, net=self.station.network,
                        ch=self.station.channel, name=self.station.name,
                        filters=self.filters)

    def add(self, trace, savetrace=False):
        """
        Adds (spectrum of) trace to spectrum.
        @type trace: L{obspy.core.trace.Trace}
        """
        self.spectrum.add(trace)
        if savetrace:
            self.savedtrace = trace


class SpectrumList(list):
    """
    List of amplitude spectra: one spectrum per station
    and per filtering sequence
    """

    def __init__(self):
        """
        @type self: list of L{SpectrumInfos}
        """
        list.__init__(self)

    def __repr__(self):
        """
        e.g., <list of 4 spectra on 2 stations>
        @rtype: str
        """
        nstat = len(set(spect.station.name for spect in self))
        s = "<list of {nspect} spectra on {nstat} stations>"
        return s.format(nspect=len(self), nstat=nstat)

    def add(self, trace, station, filters, starttime=None,
            endtime=None, savetrace=False):
        """
        Adds (spectrum of) trace to spectrum list: if station/filters already exist
        in list, (spectrum of) trace is stacked. Else, a new spectrum is appended.

        @type trace: L{obspy.core.trace.Trace}
        @type station: L{Station}
        @type filters: str
        @type starttime: L{UTCDateTime}
        @type endtime: L{UTCDateTime}
        @type savetrace: bool
        """

        # trimming trace if needed (and always working with copy!)
        trcopy = trace.copy()
        if starttime:
            trcopy.trim(starttime=starttime)
        if endtime:
            trcopy.trim(endtime=endtime)

        try:
            # looking for spectrum of station/filtering steps in list
            spectrum = next(spect for spect in self
                            if spect.station == station and spect.filters == filters)
        except StopIteration:
            # no spectrum for station/filters
            # -> appending a new SpectrumInfos instance
            spectrum_infos = SpectrumInfos(station=station, filters=filters,
                                           trace=trcopy, savetrace=savetrace)
            self.append(spectrum_infos)
        else:
            # spectrum of station/filters already exists
            # -> adding spectrum of current trace
            spectrum.add(trace=trcopy, savetrace=savetrace)

    def plot(self, smooth_window_freq=0.0):
        """
        Plots list of spectra: rows = filtering steps, columns = stations
        Plotting freq x abs(Fourier coefs)
        """

        # list of stations and filters in spectra (preserving order)
        filterslist = []
        stationlist = []
        for spect in self:
            assert isinstance(spect, SpectrumInfos)
            if spect.filters not in filterslist:
                filterslist.append(spect.filters)
            if spect.station not in stationlist:
                stationlist.append(spect.station)

        # rows = filtering steps, columns = stations: 1 station = 1 spectrum [+ 1 trace]
        nrow = len(filterslist)
        ncol = len(stationlist)

        # plot per pair (station, filters)
        plottraces = any([spect.savedtrace for spect in self])
        plotperpair = 1 if not plottraces else 2

        plt.figure()
        for ipair, (filters, station) in enumerate(it.product(filterslist, stationlist)):
            assert isinstance(filters, str)
            try:
                # looking for spectrum of station/filters
                spect = next(spect for spect in self
                             if spect.station == station and spect.filters == filters)
            except StopIteration:
                continue

            # getting freq and amplitude arrays
            spectrum = spect.spectrum
            assert isinstance(spectrum, FreqAmplSpectrum)
            freq = spectrum.freq
            ampl = np.abs(spectrum.coef)

            # smoothing amplitude spectrum (except after spectral whitening)
            if not 'white' in filters.lower():
                halfwindow = int(round(smooth_window_freq / (freq[1] - freq[0]) / 2.0))
                ampl = psutils.moving_avg(ampl, halfwindow)

            # position of current station/filters in plot
            irow = ipair / ncol + 1
            icol = ipair % ncol + 1
            pos = (irow - 1) * ncol * plotperpair + (icol - 1) * plotperpair + 1

            # plotting frequence-amplitude
            plt.subplot(nrow, ncol * plotperpair, pos)
            plt.plot(freq[100:], ampl[100:])
            plt.xlim((0.0, 0.5))
            if icol == 1:
                plt.ylabel(filters)
            if irow == 1:
                plt.title('{station} (ampl spectrum)'.format(station=spect.station.name))
            if irow == nrow:
                plt.xlabel('Frequency (Hz)')

            # plotting trace
            if plottraces and spect.savedtrace:
                tr = spect.savedtrace
                t = np.arange(0, tr.stats.npts / tr.stats.sampling_rate, tr.stats.delta)
                t /= 3600.0
                plt.subplot(nrow, ncol * plotperpair, pos + 1)
                plt.plot(t, tr.data)
                if irow == 1:
                    plt.title('{station} (last day)'.format(station=spect.station.name))
                if irow == nrow:
                    plt.xlabel('Time (h)')
        plt.show()