"""
Module related to seismic tomography
"""

import psutils
import itertools as it
import numpy as np
from scipy.optimize import curve_fit
import os
import glob
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec
from matplotlib.colors import ColorConverter
import shutil
from inspect import getargspec


# ====================================================
# parsing configuration file to import some parameters
# ====================================================
from psconfig import (
    SIGNAL_WINDOW_VMIN, SIGNAL_WINDOW_VMAX, SIGNAL2NOISE_TRAIL, NOISE_WINDOW_SIZE,
    MINSPECTSNR, MINSPECTSNR_NOSDEV, MAXSDEV, MINNBTRIMESTER, MAXPERIOD_FACTOR,
    LONSTEP, LATSTEP, CORRELATION_LENGTH, ALPHA, BETA, LAMBDA,
    FTAN_ALPHA, FTAN_VELOCITIES_STEP, PERIOD_RESAMPLE)

# ========================
# Constants and parameters
# ========================

EPS = 1.0E-6

# custom color map for seismic anomalies
# --------------------------------------
c = ColorConverter()
colors = ['black', 'red', 'gold', 'white',
          'white', 'aquamarine', 'blue', 'mediumvioletred']
values = [-1.0, -0.4, -0.1, -0.025,
          0.025, 0.1, 0.4, 1.0]
rgblist = [c.to_rgb(s) for s in colors]
reds, greens, blues = zip(*rgblist)
cdict = {}
for x, r, g, b in zip(values, reds, greens, blues):
    v = (x - min(values)) / (max(values) - min(values))
    cdict.setdefault('red', []).append((v, r, r))
    cdict.setdefault('green', []).append((v, g, g))
    cdict.setdefault('blue', []).append((v, b, b))

CMAP_SEISMIC = LinearSegmentedColormap('customseismic', cdict)

# custom color map for spatial resolution
# ---------------------------------------
colors = ['black', 'red', 'yellow', 'green', 'white']
values = [0, 0.25, 0.5, 0.75,  1.0]
rgblist = [c.to_rgb(s) for s in colors]
reds, greens, blues = zip(*rgblist)
cdict = {}
for x, r, g, b in zip(values, reds, greens, blues):
    v = (x - min(values)) / (max(values) - min(values))
    cdict.setdefault('red', []).append((v, r, r))
    cdict.setdefault('green', []).append((v, g, g))
    cdict.setdefault('blue', []).append((v, b, b))

CMAP_RESOLUTION = LinearSegmentedColormap('customresolution', cdict)
CMAP_RESOLUTION.set_bad(color='0.85')


class DispersionCurve:
    """
    Class holding a dispersion curve, i.e., velocity
    as a function of period
    """
    def __init__(self, periods, v, station1, station2,
                 minspectSNR=MINSPECTSNR, minspectSNR_nosdev=MINSPECTSNR_NOSDEV,
                 maxsdev=MAXSDEV, minnbtrimester=MINNBTRIMESTER,
                 maxperiodfactor=MAXPERIOD_FACTOR):
        """
        @type periods: iterable
        @type v: iterable
        @type station1: L{psstation.Station}
        @type station2: L{psstation.Station}
        """
        # periods and associated velocities
        self.periods = np.array(periods)
        self.v = np.array(v)
        # SNRs along periods
        self._SNRs = None
        # trimester velocities and SNRs
        self.v_trimesters = {}
        self._SNRs_trimesters = {}

        # stations
        self.station1 = station1
        self.station2 = station2

        # filter parameters
        self.minspectSNR = minspectSNR
        self.minspectSNR_nosdev = minspectSNR_nosdev
        self.maxsdev = maxsdev
        self.minnbtrimester = minnbtrimester
        self.maxperiodfactor = maxperiodfactor

    def __repr__(self):
        return 'Dispersion curve between stations {}-{}'.format(self.station1.name,
                                                                self.station2.name)

    def get_period_index(self, period):
        """
        Gets index of *period*, or raises an error if period
        is not found
        """
        iperiod = np.abs(self.periods - period).argmin()
        if np.abs(self.periods[iperiod] - period) > EPS:
            raise Exception('Cannot find period in dispersion curve')
        return iperiod

    def update_parameters(self, minspectSNR=None, minspectSNR_nosdev=None,
                          maxsdev=None, minnbtrimester=None, maxperiodfactor=None):
        """
        Updating one or more filtering parameter(s)
        """
        if not minspectSNR is None:
            self.minspectSNR = minspectSNR
        if not minspectSNR_nosdev is None:
            self.minspectSNR_nosdev = minspectSNR_nosdev
        if not maxsdev is None:
            self.maxsdev = maxsdev
        if not minnbtrimester is None:
            self.minnbtrimester = minnbtrimester
        if not maxperiodfactor is None:
            self.maxperiodfactor = maxperiodfactor

    def dist(self):
        """
        Interstation spacing (km)
        """
        return self.station1.dist(self.station2)

    def add_trimester(self, trimester_start, curve_trimester):
        """
        Adding a trimester dispersion curve.

        @type trimester_start: int
        @type curve_trimester: L{DispersionCurve}
        """
        if trimester_start in self.v_trimesters:
            raise Exception('Trimester already added')

        if np.any(curve_trimester.periods != self.periods):
            raise Exception("Wrong periods for trimester curve")

        # adding velocity adn SNR arrays of trimester
        self.v_trimesters[trimester_start] = curve_trimester.v
        self._SNRs_trimesters[trimester_start] = curve_trimester._SNRs

    def add_SNRs(self, xc, filter_alpha=FTAN_ALPHA, months=None,
                 vmin=SIGNAL_WINDOW_VMIN,
                 vmax=SIGNAL_WINDOW_VMAX,
                 signal2noise_trail=SIGNAL2NOISE_TRAIL,
                 noise_window_size=NOISE_WINDOW_SIZE):
        """
        Adding spectral SNRs at each period of the dispersion curve.
        The SNRs are calculated from the cross-correlation data
        bandpassed with narrow Gaussian filters (similar to the filter
        used in the FTAN) centered at self.periods, and width controlled
        by *filter_alpha*. (See psutils.bandpass_gaussian().)

        Parameters *vmin*, *vmax*, *signal2noise_trail*, *noise_window_size*
        control the location of the signal window and the noise window
        (see function xc.SNR()).

        @type xc: L{CrossCorrelation}
        """
        centerperiods_and_alpha = zip(self.periods, [filter_alpha] * len(self.periods))
        self._SNRs = xc.SNR(centerperiods_and_alpha=centerperiods_and_alpha,
                            months=months, vmin=vmin, vmax=vmax,
                            signal2noise_trail=signal2noise_trail,
                            noise_window_size=noise_window_size)

    def get_SNRs(self, **kwargs):
        if self._SNRs is None:
            self.add_SNRs(**kwargs)
        return self._SNRs

    def filtered_sdevs(self):
        """
        Standard dev of velocity at each period, calculated
        across trimester velocity curves. On periods at which
        std dev cannot be calculated, NaNs are returned.

        Selection criteria:
        - SNR of trimester velocity >= minspectSNR
        - nb of trimester velocities >= minnbtrimester

        @rtype: L{numpy.ndarray}
        """
        # list of arrays of trimester velocities
        trimester_vels = self.filtered_trimester_vels()

        sdevs = []
        for v_across_trimesters in zip(*trimester_vels):
            # filtering out nans from trimester velocities
            v_across_trimesters = [v for v in v_across_trimesters if not np.isnan(v)]
            if len(v_across_trimesters) >= self.minnbtrimester:
                sdev = np.std(v_across_trimesters)
            else:
                # not enough trimester velocities to estimate std dev
                sdev = np.nan
            sdevs.append(sdev)

        return np.array(sdevs) if sdevs else np.ones_like(self.periods) * np.nan

    def filtered_vels_sdevs(self):
        """
        Returns array of velocities and array of associated
        standard deviations. Velocities not passing selection
        criteria are replaced with NaNs. Where standard
        deviation cannot be estimated, NaNs are returned.

        Selection criteria:
        1) period <= distance * *maxperiodfactor*
        2) for velocities having a standard deviation associated:
           - standard deviation <= *maxsdev*
           - SNR >= *minspectSNR*
        3) for velocities NOT having a standard deviation associated:
           - SNR >= *minspectSNR_nosdev*

        @rtype: L{numpy.ndarray}, L{numpy.ndarray}
        """
        if self._SNRs is None:
            raise Exception("Spectral SNRs not defined")

        # estimating std devs, WHERE POSSIBLE (returning NaNs where not possible)
        sdevs = self.filtered_sdevs()
        has_sdev = ~np.isnan(sdevs)  # where are std devs defined?

        # Selection criteria:
        # 1) period <= distance * *maxperiodfactor*

        cutoffperiod = self.maxperiodfactor * self.dist()
        mask = self.periods <= cutoffperiod

        # 2) for velocities having a standard deviation associated:
        #    - standard deviation <= *maxsdev*
        #    - SNR >= *minspectSNR*

        mask[has_sdev] &= (sdevs[has_sdev] <= self.maxsdev) & \
                          (self._SNRs[has_sdev] >= self.minspectSNR)

        # 3) for velocities NOT having a standard deviation associated:
        #    - SNR >= *minspectSNR_nosdev*

        mask[~has_sdev] &= self._SNRs[~has_sdev] >= self.minspectSNR_nosdev

        # replacing velocities not passing the selection criteria with NaNs
        return np.where(mask, self.v, np.nan), sdevs

    def filtered_vel_sdev_SNR(self, period):
        """
        Returns a velocity, its std deviation and SNR at a given period,
        or nan if the velocity does not satisfy the criteria, or
        raises an exception if the period is not found.

        @type period: float
        @rtype: (float, float, float)
        """
        iperiod = self.get_period_index(period)

        vels, sdevs = self.filtered_vels_sdevs()
        return vels[iperiod], sdevs[iperiod], self._SNRs[iperiod]

    def filtered_trimester_vels(self):
        """
        Returns list of arrays of trimester velocities, or nan.

        Selection criteria:
        - SNR of trimester velocity >= minspectSNR
        - period <= pair distance * *maxperiodfactor*

        @rtype: list of L{numpy.ndarray}
        """
        # filtering criterion: periods <= distance * maxperiodfactor
        dist = self.station1.dist(self.station2)
        periodmask = self.periods <= self.maxperiodfactor * dist
        varrays = []
        for trimester_start, vels in self.v_trimesters.items():
            SNRs = self._SNRs_trimesters.get(trimester_start)
            if SNRs is None:
                raise Exception("Spectral SNRs not defined")
            # filtering criterion: SNR >= minspectSNR
            mask = periodmask & (SNRs >= self.minspectSNR)
            varrays.append(np.where(mask, vels, np.nan))

        return varrays


class Grid:
    """
    Class holding a 2D regular rectangular spatial grid
    """
    def __init__(self, xmin, xstep, nx, ymin, ystep, ny):
        """
        Min coords, step size and nb of points of grid
        """
        self.xmin = xmin
        self.xstep = xstep
        self.nx = int(nx)
        self.ymin = ymin
        self.ystep = ystep
        self.ny = int(ny)

    def __repr__(self):
        s = '<2D grid: x = {}...{} by {}, y = {}...{} by {}>'
        return s.format(self.xmin, self.get_xmax(), self.xstep,
                        self.ymin, self.get_ymax(), self.ystep)

    def get_xmax(self):
        return self.xmin + (self.nx - 1) * self.xstep

    def get_ymax(self):
        return self.ymin + (self.ny - 1) * self.ystep

    def bbox(self):
        """
        Bounding box: (xmin, xmax, ymin, ymax)
        @rtype: (float, float, float, float)
        """
        return self.xmin, self.get_xmax(), self.ymin, self.get_ymax()

    def n_nodes(self):
        """
        Nb of nodes on grid
        """
        return self.nx * self.ny

    def ix_iy(self, index_):
        """
        Indexes along x and y-axis of node nb *index_*
        """
        ix = np.int_(np.array(index_) / self.ny)
        iy = np.mod(np.array(index_), self.ny)
        return ix, iy

    def xy(self, index_):
        """
        Coords of node nb *index_*
        """
        index_ = np.array(index_)

        if np.any((index_ < 0) | (index_ > self.n_nodes() - 1)):
            raise Exception('Index out of bounds')

        ix, iy = self.ix_iy(index_)
        return self._x(ix), self._y(iy)

    def xy_nodes(self):
        """
        Returns coords of all nodes of grid
        """
        return self.xy(np.arange(0, self.n_nodes()))

    def xarray(self):
        return np.arange(self.xmin, self.get_xmax() + self.xstep, self.xstep)

    def yarray(self):
        return np.arange(self.ymin, self.get_ymax() + self.ystep, self.ystep)

    def index_(self, ix, iy):
        """
        Index of node (ix, iy) in grid:
        - 0 : ix=0, iy=0
        - 1 : ix=0, iy=1
        - ...
        - ny: ix=1, iy=0
        - ...
        - nx*ny-1: ix=nx-1, iy=ny-1
        """
        ix = np.array(ix)
        iy = np.array(iy)

        if np.any((ix < 0) | (ix > self.nx - 1)):
            raise Exception('ix out of bounds')
        if np.any((iy < 0) | (iy > self.ny - 1)):
            raise Exception('iy out of bounds')

        return ix * self.ny + iy

    def indexes_delaunay_triangle(self, x, y):
        """
        Indexes of the grid's nodes defining the
        Delaunay triangle around point (x, y)
        """
        # x and y indexes of bottom left neighbour
        ix = self._xindex_left_neighbour(x)
        iy = self._yindex_bottom_neighbour(y)
        np.where(ix == self.nx - 1, ix - 1, ix)
        np.where(iy == self.ny - 1, iy - 1, iy)

        xratio = (x - self._x(ix)) / self.xstep
        yratio = (y - self._y(iy)) / self.ystep

        # returning indexes of vertices of bottom right triangle
        # or upper left triangle depending on location
        index1 = self.index_(ix, iy)
        index2 = np.where(xratio >= yratio, self.index_(ix+1, iy), self.index_(ix, iy+1))
        index3 = self.index_(ix+1, iy+1)

        return index1, index2, index3

    def geodetic_dist(self, index1, index2):
        """
        Geodetic distance between nodes nb *index1* and *index2*,
        whose coodinates (x, y) are treated as (lon, lat)
        """
        lon1, lat2 = self.xy(index1)
        lon2, lat2 = self.xy(index2)
        return psutils.dist(lons1=lon1, lats1=lat2, lons2=lon2, lats2=lat2)

    def to_2D_array(self, a):
        """
        Converts a sequence-like *a* to a 2D array b[ix, iy]
        such that i is the index of node (ix, iy)
        """
        b = np.zeros((self.nx, self.ny))
        ix, iy = self.ix_iy(range(self.n_nodes()))
        b[ix, iy] = np.array(a).flatten()
        return b

    def _x(self, ix):
        """
        Returns the abscissa of node nb *ix* on x-axis
        (ix = 0 ... nx-1)
        """
        ix = np.array(ix)
        if np.any((ix < 0) | (ix > self.nx - 1)):
            raise Exception('ix out of bounds')

        return self.xmin + ix * self.xstep

    def _y(self, iy):
        """
        Returns the ordinate of node nb *iy* on y-axis
        """
        iy = np.array(iy)
        if np.any((iy < 0) | (iy > self.ny - 1)):
            raise Exception('iy out of bounds')

        return self.ymin + iy * self.ystep

    def _xindex_left_neighbour(self, x):
        """
        Returns the index (along x-axis) of the grid nodes
        closest to (and on the left of) *x*
        (Index of 1st node = 0, index of last node = nx - 1)

        @rtype: Number
        """
        x = np.array(x)
        # checking bounds
        if np.any((x < self.xmin) | (x > self.get_xmax())):
            raise Exception('x is out of bounds')

        # index of closest left node
        return np.int_((x - self.xmin) / self.xstep)

    def _yindex_bottom_neighbour(self, y):
        """
        Same as above method, along y axis

        @rtype: Number
        """
        y = np.array(y)
        # checking bounds
        if np.any((y < self.ymin) | (y > self.get_ymax())):
            raise Exception('y is out of bounds')

        # index of closest bottom node
        return np.int_((y - self.ymin) / self.ystep)


# noinspection PyShadowingNames,PyTypeChecker
class VelocityMap:
    """
    Class taking care of the inversion of velocities between
    pairs of stations, to produce a velocity map at a given
    period. The inversion procedure of Barmin et al. (2001)
    is applied.

    Attributes:
     - period      : period (s) of the velocity map
     - disp_curves : disp curves whose period's velocity is not nan
     - paths       : list of geodesic paths associated with pairs of stations
                     of dispersion curves
     - v0          : reference velocity (inverse of mean slowness, i.e.,
                     slowness implied by all observed travel-times)
     - dobs        : vector of observed data (differences observed-reference travel time)
     - Cinv        : inverse of covariance matrix of the data
     - G           : forward matrix, such that d = G.m
                     (m = parameter vector = (v0-v)/v at grid nodes)
     - density     : array of path densities at grid nodes
     - Q           : regularization matrix
     - Ginv        : inversion operator, (Gt.C^-1.G + Q)^-1.Gt
     - mopt        : vector of best-fitting parameters, Ginv.C^-1.dobs
                     = best-fitting (v0-v)/v at grid nodes
     - R           : resolution matrix, (Gt.C^-1.G + Q)^-1.Gt.C^-1.G = Ginv.C^-1.G
     - Rradius     : array of radii of the cones that best-fit each line of the
                     resolution matrix

     Note that vectors (d, m) and matrixes (Cinv, G, Q, Ginv, R) are NOT
     numpy arrays, but numpy matrixes (vectors being n x 1 matrixes). This
     means that the product operation (*) on such objects is NOT the
     element-by-element product, but the real matrix product.
    """
    def __init__(self, dispersion_curves, period, skippairs=(),
                 resolution_fit='cone', min_resolution_height=0.1,
                 showplot=False, verbose=True, **kwargs):
        """
        Initializes the velocity map at period = *period*, from
        the observed velocities in *dispersion_curves*:
        - sets up the data vector, forward matrix and regularization matrix
        - performs the tomographic inversion to estimate the best-fitting
          parameters and the resolution matrix
        - estimates the characteristic spatial resolution by fitting a cone
          to each line of the resolution matrix

        Specify pairs to be skipped (if any), as a list of pairs of stations names,
        e.g., skippairs = [('APOB', 'SPB'), ('ITAB', 'BAMB')].
        This option is useful to perform a 2-pass tomographic inversion,
        wherein pairs with a too large difference observed/predicted travel-
        time are excluded from the second pass.

        Select the type of function you want to fit to each resolution map
        with *resolution_fit*:
        - 'cone' to fit a cone, and report the cone's radius as characteristic
          resolution at each grid node in self.Rradius
        - 'gaussian' to fit a gaussian function, exp(-r/2.sigma^2), and report
          2.sigma as characteristic resolution at each grid node in self.Rradius

        Note that all resolutions in self.Rradius having a best-fitting
        cone height < *min_resolution_height* * max height will be
        discarded and set to nan.

        Append optional argument (**kwargs) to override default values:
        - minspectSNR       : min spectral SNR to retain velocity
                              (default MINSPECTSNR)
        - minspectSNR_nosdev: min spectral SNR to retain velocities without standard
                              deviation (default MINSPECTSNR_NOSDEV)
        - minnbtrimester    : min nb of trimester velocities to estimate standard
                              deviation of velocity
        - maxsdev           : max standard deviation to retain velocity (default MAXSDEV)
        - lonstep           : longitude step of grid (default LONSTEP)
        - latstep           : latitude step of grid (default LATSTEP)
        - correlation_length: correlation length of the smoothing kernel:
                                S(r,r') = exp[-|r-r'|**2 / (2 * correlation_length**2)]
                              (default value CORRELATION_LENGTH)
        - alpha             : strength of the spatial smoothing term in the penalty
                              function (default ALPHA)
        - beta              : strength of the weighted norm penalization term in the
                              penalty function (default BETA)
        - lambda_           : parameter in the damping factor of the norm penalization
                              term, such that the norm is weighted by:
                                exp(- lambda_*path_density)
                              With a value of 0.15, penalization becomes strong when
                              path density < ~20
                              With a value of 0.30, penalization becomes strong when
                              path density < ~10
                              (default LAMBDA)

        @type dispersion_curves: list of L{DispersionCurve}
        @type skippairs: list of (str, str)
        """
        self.period = period

        # reading inversion parameters
        minspectSNR = kwargs.get('minspectSNR', MINSPECTSNR)
        minspectSNR_nosdev = kwargs.get('minspectSNR_nosdev', MINSPECTSNR_NOSDEV)
        minnbtrimester = kwargs.get('minnbtrimester', MINNBTRIMESTER)
        maxsdev = kwargs.get('maxsdev', MAXSDEV)
        lonstep = kwargs.get('lonstep', LONSTEP)
        latstep = kwargs.get('latstep', LATSTEP)
        correlation_length = kwargs.get('correlation_length', CORRELATION_LENGTH)
        alpha = kwargs.get('alpha', ALPHA)
        beta = kwargs.get('beta', BETA)
        lambda_ = kwargs.get('lambda_', LAMBDA)

        if verbose:
            print "Velocities selection criteria:"
            print "- rejecting velocities if SNR < {}".format(minspectSNR)
            s = "- rejecting velocities without std dev if SNR < {}"
            print s.format(minspectSNR_nosdev)
            s = "- estimating standard dev of velocities with more than {} trimesters"
            print s.format(minnbtrimester)
            print "- rejecting velocities with standard dev > {} km/s".format(maxsdev)
            print "\nTomographic inversion parameters:"
            print "- {} x {} deg grid".format(lonstep, latstep)
            s = "- correlation length of the smoothing kernel: {} km"
            print s.format(correlation_length)
            print "- strength of the spatial smoothing term: {}".format(alpha)
            print "- strength of the norm penalization term: {}".format(beta)
            print "- weighting norm by exp(- {} * path_density)".format(lambda_)
            print

        # skipping pairs
        if skippairs:
            skippairs = [set(pair) for pair in skippairs]
            dispersion_curves = [c for c in dispersion_curves
                                 if not {c.station1.name, c.station2.name} in skippairs]

        # updating parameters of dispersion curves
        for c in dispersion_curves:
            c.update_parameters(minspectSNR=minspectSNR,
                                minspectSNR_nosdev=minspectSNR_nosdev,
                                minnbtrimester=minnbtrimester,
                                maxsdev=maxsdev)

        # valid dispersion curves (velocity != nan at period) and
        # associated interstation distances
        self.disp_curves = [c for c in dispersion_curves
                            if not np.isnan(c.filtered_vel_sdev_SNR(self.period)[0])]
        dists = np.array([c.dist() for c in self.disp_curves])

        # getting (non nan) velocities and std devs at period
        vels, sigmav, _ = zip(*[c.filtered_vel_sdev_SNR(self.period)
                                for c in self.disp_curves])
        vels = np.array(vels)
        sigmav = np.array(sigmav)
        sigmav_isnan = np.isnan(sigmav)

        # If the resolution in the velocities space is dv,
        # it means that a velocity v is actually anything between
        # v-dv/2 and v+dv/2, so the standard deviation cannot be
        # less than the standard dev of a uniform distribution of
        # width dv, which is dv / sqrt(12). Note that:
        #
        #   dv = max(dv_FTAN, dt_xc * v^2/dist),
        #
        # with dv_FTAN the intrinsic velocity discretization step
        # of the FTAN, and dt_xc the sampling interval of the
        # cross-correlation.

        dv = np.maximum(FTAN_VELOCITIES_STEP, PERIOD_RESAMPLE * vels**2 / dists)
        minsigmav = dv / np.sqrt(12)
        sigmav[~sigmav_isnan] = np.maximum(sigmav[~sigmav_isnan],
                                           minsigmav[~sigmav_isnan])

        # where std dev cannot be estimated (std dev = nan),
        # assigning 3 times the mean std dev of the period
        # following Bensen et al. (2008)
        sigmav[sigmav_isnan] = 3 * sigmav[~sigmav_isnan].mean()

        # ======================================================
        # setting up reference velocity and data vector
        # = vector of differences observed-reference travel time
        # ======================================================
        if verbose:
            print 'Setting up reference velocity (v0) and data vector (dobs)'

        # reference velocity = inverse of mean slowness
        # mean slowness = slowness implied by observed travel-times
        #               = sum(observed travel-times) / sum(intersation distances)
        s = (dists / vels).sum() / dists.sum()
        self.v0 = 1.0 / s

        # data vector
        self.dobs = np.matrix(dists / vels - dists / self.v0).T

        # inverse of covariance matrix of the data
        if verbose:
            print 'Setting up covariance matrix (C)'
        sigmad = sigmav * dists / vels**2
        self.Cinv = np.matrix(np.zeros((len(sigmav), len(sigmav))))
        np.fill_diagonal(self.Cinv, 1.0 / sigmad**2)

        # spatial grid for tomographic inversion
        lons1, lats1 = zip(*[c.station1.coord for c in self.disp_curves])
        lons2, lats2 = zip(*[c.station2.coord for c in self.disp_curves])
        lonmin = np.floor(min(lons1 + lons2))
        nlon = np.ceil((max(lons1 + lons2) - lonmin) / lonstep) + 1
        latmin = np.floor(min(lats1 + lats2))
        nlat = np.ceil((max(lats1 + lats2) - latmin) / latstep) + 1
        self.grid = Grid(lonmin, lonstep, nlon, latmin, latstep, nlat)

        # geodesic paths associated with pairs of stations of dispersion curves
        if verbose:
            print 'Calculating interstation paths'
        self.paths = []
        for curve, dist in zip(self.disp_curves, dists):
            # interpoint distance <= 1 km, and nb of points >= 100
            npts = max(np.ceil(dist) + 1, 100)
            path = psutils.geodesic(curve.station1.coord, curve.station2.coord, npts)
            self.paths.append(path)

        # ================================================
        # setting up forward matrix G, such that d = G.m
        #
        # G[i,j] = integral{w_j(r) / v0 ds} over path nb i
        # (w_j(r) = weight of node nb j on location r)
        # ================================================
        G = np.zeros((len(self.paths), self.grid.n_nodes()))
        if verbose:
            print 'Setting up {} x {} forward matrix (G)'.format(*G.shape)
        for ipath, path in enumerate(self.paths):

            # for each point M along the path (1) we determine the Delaunay
            # triangle ABC that encloses M, (2) we locally define a cartesian
            # system on the plane ABC, (3) we locate M' (the projection of M
            # on the plane ABC) and (4) we attribute weights to A, B, C
            # corresponding to the three-point linear interpolation of A, B,
            # C at point M'.

            lon_M, lat_M = path[:, 0], path[:, 1]
            xyzM = psutils.geo2cartesian(lon_M, lat_M)

            # indexes, geographic coordinates and cartesian coordinates
            # (on unit sphere) of grid nodes of Delaunay triangle ABC
            # enclosing M
            iA, iB, iC = self.grid.indexes_delaunay_triangle(lon_M, lat_M)
            lonlatA, lonlatB, lonlatC = [self.grid.xy(index_) for index_ in (iA, iB, iC)]
            xyzA, xyzB, xyzC = [psutils.geo2cartesian(lon, lat)
                                for lon, lat in (lonlatA, lonlatB, lonlatC)]

            # projection of M on the plane ABC
            xyzMp = psutils.projection(xyzM, xyzA, xyzB, xyzC)

            # weights of nodes A, B, C in linear interpolation =
            # barycentric coordinates of M' in triangle ABC
            wA, wB, wC = psutils.barycentric_coords(xyzMp, xyzA, xyzB, xyzC)

            # attributing weights to grid nodes along path:
            # w[j, :] = w_j(r) = weights of node j along path
            nM = path.shape[0]
            w = np.zeros((self.grid.n_nodes(), nM))
            w[iA, range(nM)] = wA
            w[iB, range(nM)] = wB
            w[iC, range(nM)] = wC

            # ds = array of infinitesimal distances along path
            ds = psutils.dist(lons1=lon_M[:-1], lats1=lat_M[:-1],
                              lons2=lon_M[1:], lats2=lat_M[1:])

            # integrating w_j(r) / v0 along path using trapeze formula
            G[ipath, :] = np.sum(0.5 * (w[:, :-1] + w[:, 1:]) / self.v0 * ds, axis=-1)

        self.G = np.matrix(G)

        # path densities around grid's nodes
        if verbose:
            print "Calculating path densities"
        self.density = self.path_density()

        # =================================================================
        # setting up regularization matrix Q = Ft.F + Ht.H
        #
        # F[i,j] = alpha * | 1 (1 - S(ri,ri) according to me!)    if i = j
        #                  | -S(ri,rj) / sum{S(ri,rj')} over j']  if i!= j
        #
        # H[i,j] = beta * | exp[-lambda * path_density(ri)]      if i = j
        #                 | 0                                    if i!= j
        #
        # with S(.,.) the smoothing kernel and ri the locations grid nodes
        # =================================================================

        # setting up distance matrix:
        # dists[i,j] = distance between nodes nb i and j
        dists = np.zeros((self.grid.n_nodes(), self.grid.n_nodes()))

        if verbose:
            print "Setting up {} x {} regularization matrix (Q)".format(*dists.shape)

        # indices of the upper right triangle of distance matrix
        # = (array of index #1, array of index #2)
        i_upper, j_upper = np.triu_indices_from(dists)
        lons_i, lats_i = self.grid.xy(i_upper)
        lons_j, lats_j = self.grid.xy(j_upper)
        # distance matrix (upper triangle)
        dists[i_upper, j_upper] = psutils.dist(lons1=lons_i, lats1=lats_i,
                                               lons2=lons_j, lats2=lats_j)
        # symmetrizing distance matrix (works because diagonal elts = 0)
        dists += dists.T

        # setting up smoothing kernel:
        # S[i,j] = K * exp[-|ri-rj|**2 / (2 * CORRELATION_LENGTH**2)]
        S = np.exp(- dists**2 / (2 * correlation_length**2))
        S /= S.sum(axis=-1)  # normalization

        # setting up spatial regularization matrix F
        F = np.matrix(-S)

        # F[i,i] = 1 according to Barmin et al.
        # F[i,i] = 1 - S[i,i] according to my calculation!
        # -> difference is negligible??

        F[np.diag_indices_from(F)] = 1
        F *= alpha

        # setting up regularization matrix Q
        # ... Ft.F part
        Q = F.T * F
        # ... Ht.H part
        for i, path_density in enumerate(self.density):
            Q[i, i] += beta**2 * np.exp(-2 * lambda_ * path_density)

        self.Q = Q

        # ===========================================================
        # setting up inversion operator Ginv = (Gt.C^-1.G + Q)^-1.Gt,
        # estimating model and setting up resolution matrix R =
        # Ginv.C^-1.G
        # ===========================================================

        # inversion operator
        if verbose:
            print "Setting up inversion operator (Ginv)"
        self.Ginv = (self.G.T * self.Cinv * self.G + self.Q).I * self.G.T

        # vector of best-fitting parameters
        if verbose:
            print "Estimating best-fitting parameters (mopt)"
        self.mopt = self.Ginv * self.Cinv * self.dobs

        # resolution matrix
        if verbose:
            print "Setting up {0} x {0} resolution matrix (R)".format(self.G.shape[1])
        self.R = self.Ginv * self.Cinv * self.G

        # ===========================================================
        # Estimating spatial resolution at each node of the grid,
        # Rradius.
        #
        # The i-th row of the resolution matrix, R[i,:], contains the
        # resolution map associated with the i-th grid noe, that is,
        # the estimated model we would get if there were only a point
        # velocity anomaly at node nb i. So a cone centered on node
        # nb i is fitted to the resolution map, and its radius gives
        # an indication of the spatial resolution at node nb i (i.e.,
        # the minimum distance at which two point anomalies can be
        # resolved)
        # ===========================================================

        if verbose:
            print "Estimation spatial resolution (Rradius)"

        self.Rradius = np.zeros(self.grid.n_nodes())
        heights = np.zeros(self.grid.n_nodes())

        for i, Ri in enumerate(np.array(self.R)):
            lon0, lat0 = self.grid.xy(i)

            # best-fitting cone at point (lon0, lat0)

            # Function returning the height of cone of radius *r0*
            # and peak *z0*, at a point located *r* km away from
            # the cone's center
            if resolution_fit.lower().strip() == 'cone':
                def cone_height(r, z0, r0):
                    """
                    Cone
                    """
                    return np.where(r < r0, z0 * (1 - r / r0), 0.0)
            elif resolution_fit.lower().strip() == 'gaussian':
                def cone_height(r, z0, r0):
                    """
                    Gaussian function
                    """
                    sigma = r0 / 2.0
                    return z0 * np.exp(- r**2 / (2 * sigma**2))
            else:
                s = "Unknown function to fit resolution: '{}'"
                raise Exception(s.format(resolution_fit))

            # distances between nodes and cone's center (lon0, lat0)
            lonnodes, latnodes = self.grid.xy_nodes()
            n = self.grid.n_nodes()
            rdata = psutils.dist(lons1=lonnodes, lats1=latnodes,
                                 lons2=n*[lon0], lats2=n*[lat0])

            # best possible resolution *rmin* = 2 * inter-node distance
            # -> estimating *rmin* along the meridian crossing the cone's
            #    center (conservative choice as it yields the largest
            #    possible value)
            d2rad = np.pi / 180.0
            rmin = 2 * d2rad * 6371.0 * max(self.grid.xstep * np.cos(lat0 * d2rad),
                                            self.grid.ystep)

            # fitting the above function to observed heights along nodes,
            # in array abs(Ri)
            popt, _ = curve_fit(f=cone_height, xdata=rdata, ydata=np.abs(Ri),
                                p0=[1, 2*rmin], maxfev=10000)
            z0, r0 = popt

            # reslution cannot be better than *rmin*
            r0 = max(rmin, r0)

            # appending spatial resolution to array
            self.Rradius[i] = r0
            heights[i] = z0

        self.Rradius[heights < heights.max() * min_resolution_height] = np.nan

        if showplot:
            # potting maps of velocity perturbation,
            # path density and resolution
            _ = self.plot()

    def __repr__(self):
        """
        E.g., "<Velocity map at period = 10 s>"
        """
        return '<Velocity map at period = {} s>'.format(self.period)

    def path_density(self, window=(LONSTEP, LATSTEP)):
        """
        Returns the path density, that is, on each node of the
        grid, the number of paths that cross the rectangular
        cell of size (window[0], window[1]) centered on
        the node.
        """
        # initializing path density
        density = np.zeros(self.grid.n_nodes())

        # coordinates of grid nodes and associated windows
        lons_nodes, lats_nodes = self.grid.xy_nodes()
        lons_min = np.expand_dims(lons_nodes - window[0] / 2.0, axis=-1)
        lons_max = np.expand_dims(lons_nodes + window[0] / 2.0, axis=-1)
        lats_min = np.expand_dims(lats_nodes - window[1] / 2.0, axis=-1)
        lats_max = np.expand_dims(lats_nodes + window[1] / 2.0, axis=-1)

        for path in self.paths:
            lons_path, lats_path = path[:, 0], path[:, 1]
            # are points of paths in windows?
            # 1st dim = grid nodes; 2nd dim = points along path
            points_in_windows = (lons_path >= lons_min) & (lons_path <= lons_max) & \
                                (lats_path >= lats_min) & (lats_path <= lats_max)
            density += np.any(points_in_windows, axis=-1)

        return density

    def traveltime_residuals(self):
        """
        Returns the residual between observed-predicted travel times
        at each pair of stations:

          residuals = observed - predicted travel-time,
                    = dobs - dpred,
          with dpred = G.mopt

        @rtype: L{matrix}
        """
        return self.dobs - self.G * self.mopt

    def checkerboard_func(self, vmid, vmin, vmax, squaresize, shape='cos'):
        """
        Returns a checkerboard function, f(lons, lats), whose background
        value is *vmid*, and alternating min/max values are *vmin* and
        *vmax*. The centers of the anomalies are separated by *squaresize*
        (in km), and their shape is either 'gaussian' or 'cos'.

        @rtype: function
        """
        # converting square size from km to degrees
        d2rad = np.pi / 180.0
        midlat = 0.5 * (self.grid.ymin + self.grid.get_ymax())
        latwidth = squaresize / 6371.0 / d2rad
        lonwidth = squaresize / (6371.0 * np.cos(midlat * d2rad)) / d2rad

        # Basis function defining an anomaly of
        # unit height centered at (*lon0*, *lat0*).
        if shape.lower().strip() == 'gaussian':
            def basis_func(lons, lats, lon0, lat0):
                """
                Gausian anomaly , with sigma-parameter such that 3 sigma
                is the distance between the center and the border of
                the square, that is, half the distance between 2
                centers.
                """
                n = len(lons)
                r = psutils.dist(lons1=lons, lats1=lats, lons2=n*[lon0], lats2=n*[lat0])
                sigma = squaresize / 6.0
                return np.exp(- r**2 / (2 * sigma**2))
        elif shape.lower().strip() == 'cos':
            def basis_func(lons, lats, lon0, lat0):
                """
                Cosinus anomaly
                """
                x = (lons - lon0) / lonwidth
                y = (lats - lat0) / latwidth
                outside_square = (np.abs(x) >= 0.5) | (np.abs(y) >= 0.5)
                return np.where(outside_square, 0.0, np.cos(np.pi*x) * np.cos(np.pi*y))
        else:
            raise Exception("Unknown shape anomaly: " + shape)

        # coordinates of the center of the anomalies
        startlon = self.grid.xmin + lonwidth / 2.0
        stoplon = self.grid.get_xmax() + lonwidth
        centerlons = list(np.arange(startlon, stoplon, lonwidth))
        startlat = self.grid.ymin + latwidth / 2.0
        stoplat = self.grid.get_ymax() + latwidth
        centerlats = list(np.arange(startlat, stoplat, latwidth))
        centerlonlats = list(it.product(centerlons, centerlats))

        # factors by which multiply the basis function associated
        # with each center (to alternate lows and highs)
        polarities = [(centerlons.index(lon) + centerlats.index(lat)) % 2
                      for lon, lat in centerlonlats]
        factors = np.where(np.array(polarities) == 1, vmax - vmid, vmin - vmid)

        def func(lons, lats):
            """
            Checkboard function: sum of the basis functions along
            the centers defined above, times the high/low factor,
            plus background velocity.
            """
            lowhighs = [f * basis_func(lons, lats, lon0, lat0) for f, (lon0, lat0)
                        in zip(factors, centerlonlats)]
            return vmid + sum(lowhighs)

        return func

    def checkerboard_test(self, vmid, vmin, vmax, squaresize, **kwargs):
        """
        Generates synthetic data (travel time perturbations),
        dsynth, from a checkerboard model of velocities, and
        performs a tomographic inversion on them:

          m = (Gt.C^-1.G + Q)^-1.Gt.C^-1.dsynth
            = Ginv.C^-1.dsynth

        Returns the vector of best-fitting parameters, m.

        @rtype: L{matrix}
        """

        # checkerboard function
        f_checkerboard = self.checkerboard_func(vmid, vmin, vmax, squaresize, **kwargs)

        # setting up vector of synthetic data
        dsynth = np.zeros_like(self.dobs)
        for d, path, curve in zip(dsynth, self.paths, self.disp_curves):
            # array of infinitesimal distances along path
            lons, lats = path[:, 0], path[:, 1]
            ds = psutils.dist(lons1=lons[:-1], lats1=lats[:-1],
                              lons2=lons[1:], lats2=lats[1:])

            # velocities along path
            v = f_checkerboard(lons, lats)

            # travel time = integral[ds / v]
            t = np.sum(ds * 0.5 * (1.0 / v[:-1] + 1.0 / v[1:]))

            # synthetic data = travel time - ref travel time
            d[...] = t - curve.dist() / vmid

        # inverting synthetic data
        m = self.Ginv * self.Cinv * dsynth
        return m

    def plot(self, xsize=20, title=None, showplot=True, outfile=None, **kwargs):
        """
        Plots velocity perturbation, path density
        and spatial resolution, and returns the figure.

        Additional keyword args in *kwargs* are sent to
        self.plot_velocity(), self.plot_pathdensity()
        and self.plot_resolution(), when applicable

        @rtype: L{matplotlib.figure.Figure}
        """
        # bounding box
        bbox = self.grid.bbox()
        aspectratio = (bbox[3] - bbox[2]) / (bbox[1] - bbox[0])
        figsize = (xsize, aspectratio * xsize / 3.0 + 2)
        fig = plt.figure(figsize=figsize)

        # layout
        gs = gridspec.GridSpec(1, 3, wspace=0.0, hspace=0.0)

        # plotting velocity perturbation
        ax = fig.add_subplot(gs[0, 0])
        subkwargs = {'ax': ax, 'plot_title': False}
        # sending additional arguments (when applicable)
        subkwargs.update({k: kwargs[k] for k in getargspec(self.plot_velocity).args
                         if k in kwargs})
        self.plot_velocity(**subkwargs)

        # plotting path density
        ax = fig.add_subplot(gs[0, 1])
        subkwargs = {'ax': ax, 'plot_title': False, 'stationlabel': True}
        # sending additional arguments (when applicable)
        subkwargs.update({k: kwargs[k] for k in getargspec(self.plot_pathdensity).args
                         if k in kwargs})
        self.plot_pathdensity(**subkwargs)

        # plotting spatial resolution
        ax = fig.add_subplot(gs[0, 2])
        subkwargs = {'ax': ax, 'plot_title': False}
        # sending additional arguments (when applicable)
        subkwargs.update({k: kwargs[k] for k in getargspec(self.plot_resolution).args
                         if k in kwargs})
        self.plot_resolution(**subkwargs)

        # fig title
        if not title:
            # default title if not given
            title = u'Period = {} s, {} paths'
            title = title.format(self.period, len(self.paths))
        fig.suptitle(title, fontsize=16)

        gs.tight_layout(fig, rect=[0, 0, 1, 0.95])

        # saving figure
        if outfile:
            if os.path.exists(outfile):
                # backup
                shutil.copyfile(outfile, outfile + '~')
            fig.set_size_inches(figsize)
            fig.savefig(outfile, dpi=300)

        # showing figure
        if showplot:
            fig.show()

        return fig

    def plot_pathdensity(self, ax=None, xsize=10, plotdensity=True, plotpaths=True,
                         stationlabel=False, plot_title=True, showgrid=False,
                         highlight_residuals_gt=None):
        """
        Plots path density and/or interstation paths.

        Paths for which the residual observed/predicted travel-time
        is greater than *highlight_residuals_gt* (if defined) are
        highlighted as bold lines.
        """
        # bounding box
        bbox = self.grid.bbox()

        # creating figure if not given as input
        fig = None
        if not ax:
            aspectratio = (bbox[3] - bbox[2]) / (bbox[1] - bbox[0])
            # xzise has not effect if axes are given as input
            fig = plt.figure(figsize=(xsize, aspectratio * xsize), tight_layout=True)
            ax = fig.add_subplot(111)

        # plotting coasts and tectonic provinces
        psutils.basemap(ax=ax, labels=False, fill=not plotdensity, bbox=bbox)

        if plotdensity:
            # plotting path density
            d = self.grid.to_2D_array(self.density)
            extent = (self.grid.xmin, self.grid.get_xmax(),
                      self.grid.ymin, self.grid.get_ymax())
            m = ax.imshow(d.transpose(), origin='bottom', extent=extent,
                          interpolation='bicubic',
                          cmap=pathdensity_colormap(dmax=d.max()))
            c = plt.colorbar(m, ax=ax, orientation='horizontal', pad=0.1,
                             ticks=range(0, int(d.max()) + 1, 5))
            c.set_label('Path density')

        if plotpaths:
            # residuals observed/predicted travel-times
            res = self.traveltime_residuals() if highlight_residuals_gt else []

            # plotting paths
            for i, path in enumerate(self.paths):
                x, y = zip(*path)
                linestyle = {'color': 'grey', 'lw': 0.5}
                if highlight_residuals_gt and abs(float(res[i])) > highlight_residuals_gt:
                    # highlighting line as the travel-time error is > threshold
                    linestyle = {'color': 'black', 'lw': 1.5}
                ax.plot(x, y, '-', **linestyle)

        if showgrid:
            # plotting grid
            x, y = self.grid.xy_nodes()
            ax.plot(x, y, '+')

        # plotting stations
        self._plot_stations(ax, stationlabel=stationlabel)

        # formatting axes
        ax.set_xlim(bbox[:2])
        ax.set_ylim(bbox[2:])
        if plot_title:
            ax.set_title(u'Period = {} s, {} paths'.format(self.period, len(self.paths)))

        if fig:
            fig.show()

    def plot_velocity(self, ax=None, xsize=10, perturbation=False, plot_title=True,
                      vscale=None):
        """
        Plots velocity or perturbation relative to mean velocity
        (which is not necessarily the reference velocity)
        """
        # bounding box
        bbox = self.grid.bbox()

        # creating figure if not given as input
        fig = None
        if not ax:
            aspectratio = (bbox[3] - bbox[2]) / (bbox[1] - bbox[0])
            # xzise has not effect if axes are given as input
            fig = plt.figure(figsize=(xsize, aspectratio * xsize))
            ax = fig.add_subplot(111)

        # plotting coasts and tectonic provinces
        psutils.basemap(ax=ax, labels=False, fill=False, bbox=bbox)

        # plotting stations
        self._plot_stations(ax, stationlabel=False)

        # velocities on grid: m = (v0 - v) / v, so v = v0 / (1 + m)
        v = self.grid.to_2D_array(self.v0 / (1 + self.mopt))
        vmean = v.mean()
        if perturbation:
            # plotting % perturbation relative to mean velocity
            v = 100 * (v - vmean) / vmean

        if not vscale and perturbation:
            # symetric scale
            maxdv = np.abs(v).max()
            vscale = (-maxdv, maxdv)
        elif not vscale and not perturbation:
            # scale centered on mean velocity
            maxdv = np.abs(v - vmean).max()
            vscale = (vmean - maxdv, vmean + maxdv)

        extent = (self.grid.xmin, self.grid.get_xmax(),
                  self.grid.ymin, self.grid.get_ymax())
        m = ax.imshow(v.transpose(), origin='bottom', extent=extent,
                      interpolation='bicubic', cmap=CMAP_SEISMIC,
                      vmin=vscale[0], vmax=vscale[1])
        c = plt.colorbar(m, ax=ax, orientation='horizontal', pad=0.1)
        c.set_label('Velocity perturbation (%)' if perturbation else 'Velocity (km/s)')

        # formatting axes
        ax.set_xlim(bbox[:2])
        ax.set_ylim(bbox[2:])
        if plot_title:
            ax.set_title(u'Period = {} s, {} paths'.format(self.period, len(self.paths)))

        if fig:
            fig.show()

    def plot_resolution(self, ax=None, xsize=10, plot_title=True):
        """
        Plots resolution map
        """
        # bounding box
        bbox = self.grid.bbox()

        # creating figure if not given as input
        fig = None
        if not ax:
            aspectratio = (bbox[3] - bbox[2]) / (bbox[1] - bbox[0])
            # xzise has not effect if axes are given as input
            fig = plt.figure(figsize=(xsize, aspectratio * xsize), tight_layout=True)
            ax = fig.add_subplot(111)

        # plotting coasts and tectonic provinces
        psutils.basemap(ax=ax, labels=False, fill=False, bbox=bbox)

        # plotting stations
        self._plot_stations(ax, stationlabel=False)

        # plotting spatial resolution
        r = self.grid.to_2D_array(self.Rradius)
        extent = (self.grid.xmin, self.grid.get_xmax(),
                  self.grid.ymin, self.grid.get_ymax())
        m = ax.imshow(r.transpose(), origin='bottom', extent=extent,
                      interpolation='bicubic',
                      cmap=CMAP_RESOLUTION)
        c = plt.colorbar(m, ax=ax, orientation='horizontal', pad=0.1)
        c.set_label('Spatial resolution (km)')

        # formatting axes
        ax.set_xlim(bbox[:2])
        ax.set_ylim(bbox[2:])
        if plot_title:
            ax.set_title(u'Period = {} s, {} paths'.format(self.period, len(self.paths)))

        if fig:
            fig.show()

    def plot_checkerboard(self, vmid, vmin, vmax, squaresize, axes=None, xsize=10,
                          **kwargs):
        """
        Plots checkboard model and reconstructed checkerboard
        """
        # checkerboard test
        m = self.checkerboard_test(vmid, vmin, vmax, squaresize, **kwargs)
        v = self.grid.to_2D_array(vmid / (1 + m))
        dv = 100 * (v - vmid) / vmid

        # bounding box
        bbox = self.grid.bbox()

        # creating figure if not given as input
        fig = None
        if not axes:
            aspectratio = (bbox[3] - bbox[2]) / (bbox[1] - bbox[0])
            # xzise has not effect if axes are given as input
            fig = plt.figure(figsize=(xsize, aspectratio * xsize), tight_layout=True)
            axes = [fig.add_subplot(121), fig.add_subplot(122)]

        ims = []

        # checkerboard model
        checkerboard_func = self.checkerboard_func(vmid, vmin, vmax, squaresize, **kwargs)
        lons, lats = self.grid.xy_nodes()
        a = self.grid.to_2D_array(checkerboard_func(lons, lats))
        extent = (self.grid.xmin, self.grid.get_xmax(),
                  self.grid.ymin, self.grid.get_ymax())
        im = axes[0].imshow(a.transpose(),
                            origin='bottom', extent=extent,
                            interpolation='bicubic',
                            vmin=vmin, vmax=vmax,
                            cmap=CMAP_SEISMIC)
        ims.append(im)

        # reconstructed checkerboard
        extent = (self.grid.xmin, self.grid.get_xmax(),
                  self.grid.ymin, self.grid.get_ymax())
        im = axes[1].imshow(dv.transpose(),
                            origin='bottom', extent=extent,
                            interpolation='bicubic',
                            vmin=-np.abs(dv).max(),
                            vmax=np.abs(dv).max(),
                            cmap=CMAP_SEISMIC)
        ims.append(im)

        for ax, im in zip(axes, ims):
            # coasts and tectonic provinces
            psutils.basemap(ax=ax, labels=False, fill=False, bbox=bbox)

            # stations
            self._plot_stations(ax, stationlabel=False)

            # color bar
            c = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
            c.set_label('km/s' if ax is axes[0] else '% perturbation')

            # limits
            ax.set_xlim(bbox[:2])
            ax.set_ylim(bbox[2:])

        if fig:
            fig.show()

    def _plot_stations(self, ax, stationlabel):
        """
        Plots stations on map
        """
        # plotting stations
        xylabels = [c.station1.coord + (c.station1.name,) for c in self.disp_curves] + \
                   [c.station2.coord + (c.station2.name,) for c in self.disp_curves]
        xlist, ylist, labels = zip(*list(set(xylabels)))
        ax.plot(xlist, ylist, '^', color='k', ms=10, mfc='w', mew=1)

        if not stationlabel:
            return

        # stations label
        for x, y, label in zip(xlist, ylist, labels):
            ax.text(x, y, label, ha='center', va='bottom', fontsize=10, weight='bold')


def pathdensity_colormap(dmax):
    """
    Builds a colormap for path density (d) varying from
    0 to *dmax*:
    - white for d = 0
    - blue to green for 1 <= d <= 5
    - green to red for 5 <= d <= 10
    - red to black for 10 <= d <= dmax
    """
    dmax = max(dmax, 11)
    x1 = 1.0 / dmax
    x2 = 5.0 / dmax
    x3 = 10.0 / dmax
    cdict = {'red': ((0, 1, 1), (x1, 0, 0), (x2, 0, 0), (x3, 1, 1), (1, 0, 0)),
             'green': ((0, 1, 1), (x1, 0, 0), (x2, 1, 1), (x3, 0, 0), (1, 0, 0)),
             'blue': ((0, 1, 1), (x1, 1, 1), (x2, 0, 0), (x3, 0, 0), (1, 0, 0))}
    return LinearSegmentedColormap('tmp', cdict)


if __name__ == '__main__':
    # importig dir of FTAN results
    from psconfig import FTAN_DIR

    # loading dispersion curves
    flist = sorted(glob.glob(os.path.join(FTAN_DIR, 'FTAN*.pickle*')))
    print 'Select file containing dispersion curves:'
    print '\n'.join('{} - {}'.format(i, os.path.basename(f)) for i, f in enumerate(flist))
    pickle_file = flist[int(raw_input('\n'))]
    f = open(pickle_file, 'rb')
    curves = pickle.load(f)
    f.close()
    print "Dispersion curves stored in variable 'curves'"