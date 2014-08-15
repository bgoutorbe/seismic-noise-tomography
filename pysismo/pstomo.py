"""
Module related to seismic tomography
"""

import psutils
import numpy as np
from scipy.optimize import curve_fit
import os
import glob
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ====================================
# Default dispersion curves parameters
# ====================================

# min spectral SNR to retain [trimester] velocity
# todo: lower min SNR for trimester FTAN??
MINSPECTSNR = 10
# min spectral SNR to retain velocity if no std dev
MINSPECTSNR_NOSDEV = 15
# min sdt dev (km/s) to retain velocity
MINSDEV = 0.1
# min nb of trimesters to estimate std dev
MINNBTRIMESTER = 4
# max period = *MAXPERIOD_FACTOR* * pair distance
MAXPERIOD_FACTOR = 1.0 / 12.0

# internode spacing of grid for tomographic inversion
LONSTEP = 1  # 0.5
LATSTEP = 1  # 0.5

# =========================
# regularization parameters
# =========================

# correlation length of the smoothing kernel:
# S(r,r') = exp[-|r-r'|**2 / (2 * CORRELATION_LENGTH**2)]
CORRELATION_LENGTH = 200

# strength of the spatial smoothing term (alpha) and the
# weighted norm penalization term (beta) relatively to
# the strength of the misfit, in the penalty function
ALPHA = 800
BETA = 100

# lambda parameter, such that the norm is weighted by
# exp(-lambda.path_density) in the the norm penalization
# term of the penalty function
LAMBDA = 0.147

EPS = 1.0E-6


class DispersionCurve:
    """
    Class holding a dispersion curve, i.e., velocity
    as a function of period
    """
    def __init__(self, periods, v, station1, station2,
                 minspectSNR=MINSPECTSNR, minspectSNR_nosdev=MINSPECTSNR_NOSDEV,
                 minsdev=MINSDEV, minnbtrimester=MINNBTRIMESTER,
                 maxperiodfactor=MAXPERIOD_FACTOR):
        """
        @type periods: L{ndarray}
        @type v: L{ndarray}
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
        self.maxsdev = minsdev
        self.minnbtrimester = minnbtrimester
        self.maxperiodfactor = maxperiodfactor

    def __repr__(self):
        return 'Dispersion curve between stations {}-{}'.format(self.station1.name,
                                                                self.station2.name)

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

    def add_SNRs(self, xc, relfreqwin=0.2, months=None):
        """
        Adding spectral SNRs at each period of the dispersion curve.
        The SNRs are calculated from the cross-correlation data
        bandpassed along windows centered on freqs = 1 / periods,
        with widths = +/- *relfreqwin* x freqs

        @type xc: L{CrossCorrelation}
        """
        bands = [(T / (1.0 + relfreqwin), T / (1.0 - relfreqwin)) for T in self.periods]
        self._SNRs = xc.SNR(bands, months=months)

    def filtered_sdevs(self):
        """
        Standard dev of velocity at each period, calculated
        across trimester velocity curves. On periods at which
        std dev cannot be calculated, nan are returned.

        Filtering criteria:
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
        Returns array of velocities and array of (std dev or nan).

        Filtering criteria:
        - period <= pair distance * *maxperiodfactor*
        AND
        - SNR of velocity >= minspectSNR_nosdev
        OR
        - SNR of velocity >= minspectSNR
        - std dev != nan and <= max std dev

        @rtype: L{numpy.ndarray}, L{numpy.ndarray}
        """
        if self._SNRs is None:
            raise Exception("Spectral SNRs not defined")

        # std devs
        sdevs = self.filtered_sdevs()

        # filtering criteria:
        # - periods <= distance * maxperiodfactor
        mask = self.periods <= self.maxperiodfactor * self.station1.dist(self.station2)
        # - SNR >= minspectSNR_nosdev or
        #   SNR >= minspectSNR and std dev <= min std dev
        mask &= (self._SNRs >= self.minspectSNR_nosdev) | \
                ((self._SNRs >= self.minspectSNR) &
                 np.where(np.isnan(sdevs), False, sdevs <= self.maxsdev))

        return np.where(mask, self.v, np.nan), sdevs

    def filtered_vel_sdev_SNR(self, period):
        """
        Returns a velocity, its std deviation and SNR at a given period,
        or nan if the velocity does not satisfy the criteria, or
        raises an exception if the period is not found.

        @type period: float
        @rtype: (float, float, float)
        """
        iperiod = np.abs(self.periods - period).argmin()
        if np.abs(self.periods[iperiod] - period) > EPS:
            raise Exception('Cannot find period in disperion curve')

        vels, sdevs = self.filtered_vels_sdevs()
        return vels[iperiod], sdevs[iperiod], self._SNRs[iperiod]

    def filtered_trimester_vels(self):
        """
        Returns list of arrays of trimester velocities, or nan.

        Filtering criteria:
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
        Converts a 1D array a[i] to a 2D array b[ix, iy]
        such that i is the index of node (ix, iy)
        """
        b = np.zeros((self.nx, self.ny))
        ix, iy = self.ix_iy(range(self.n_nodes()))
        b[ix, iy] = a
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


class VelocityMap:
    """
    Class taking care of the inversion of velocities between
    pairs of stations, to produce a velocity map at a given
    period. The inversion procedure of Barmin et al. (2001)
    is applied.

    Attributes:
     - period      : period (s) of the velocity map
     - disp_curves : disp curves whose period's velocity is not nan
     - v0          : reference velocity
     - d           : data array (differences observed-reference travel time)
     - Cinv        : inverse of covariance matrix of the data
     - G           : forward operator, such that d = G.m
                     (m = parameter array = (v0-v)/v0 at grid nodes)
     - density     : array of path densities at grid nodes
     - Q           : regularization matrix
     - Ginv        : inversion operator, (Gt.C^-1.G + Q)^-1.Gt
     - m           : estimated model, Ginv.C^-1.d
     - R           : resolution matrix, (Gt.C^-1.G + Q)^-1.Gt.C^-1.G = Ginv.C^-1.G
     - Rradius     : radii of the cones that best-fit each line of the resolution
                     matrix
    """
    def __init__(self, dispersion_curves, period, verbose=True):
        """
        @type dispersion_curves: list of L{DispersionCurve}
        """
        self.period = period

        # valid dispersion curves (velocity != nan at period)
        self.disp_curves = [c for c in dispersion_curves
                            if not np.isnan(c.filtered_vel_sdev_SNR(self.period)[0])]

        # getting (non nan) velocities and std devs at period
        vels, sigmav, _ = zip(*[c.filtered_vel_sdev_SNR(self.period)
                                for c in self.disp_curves])
        vels = np.array(vels)
        sigmav = np.array(sigmav)

        # where std dev cannot be estimated (std dev = nan),
        # assigning 3 times the mean std dev of the period
        # following Bensen et al. (2008)
        sigmav[np.isnan(sigmav)] = 3 * sigmav[-np.isnan(sigmav)].mean()

        # reference model = mean of velocities at period
        self.v0 = vels.mean()

        # =====================================================
        # setting up data array
        # = array of differences observed-reference travel time
        # =====================================================
        if verbose:
            print 'Setting up data array (d)'
        lons1, lats1 = zip(*[c.station1.coord for c in self.disp_curves])
        lons2, lats2 = zip(*[c.station2.coord for c in self.disp_curves])
        dists = psutils.dist(lons1=lons1, lats1=lats1, lons2=lons2, lats2=lats2)
        self.d = dists / vels - dists / self.v0

        # inverse of covariance matrix of the data
        if verbose:
            print 'Setting up covariance matrix (C)'
        sigmad = sigmav * dists / vels**2
        self.Cinv = np.matrix(np.zeros((len(sigmav), len(sigmav))))
        np.fill_diagonal(self.Cinv, 1.0 / sigmad**2)

        # spatial grid for tomographic inversion
        lonmin = np.floor(min(lons1 + lons2))
        nlon = np.ceil((max(lons1 + lons2) - lonmin) / LONSTEP) + 1
        latmin = np.floor(min(lats1 + lats2))
        nlat = np.ceil((max(lats1 + lats2) - latmin) / LATSTEP) + 1
        self.grid = Grid(lonmin, LONSTEP, nlon, latmin, LATSTEP, nlat)

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
        S = np.exp(-dists**2 / (2 * CORRELATION_LENGTH**2))
        S /= S.sum(axis=-1)  # normalization

        # setting up spatial regularization matrix F
        F = -S

        # F[i,i] = 1 according to Barmin et al.
        # F[i,i] = 1 - S[i,i] according to my calculation!
        # -> difference is negligible??

        F[np.diag_indices_from(F)] = 1
        F *= ALPHA

        # setting up regularization matrix Q
        # ... Ft.F part
        Q = np.dot(F.T, F)
        # ... Ht.H part
        Q[np.diag_indices_from(Q)] += BETA**2 * np.exp(-2 * LAMBDA * self.density)

        self.Q = np.matrix(Q)

        # ===========================================================
        # setting up inversion operator Ginv = (Gt.C^-1.G + Q)^-1.Gt,
        # estimating model and setting up resolution matrix R =
        # Ginv.C^-1.G
        # ===========================================================

        # inversion operator
        if verbose:
            print "Setting up inversion operator (Ginv)"
        self.Ginv = (self.G.T * self.Cinv * self.G + self.Q).I * self.G.T

        # estimated model
        if verbose:
            print "Estimating model (m)"
        m = self.Ginv * self.Cinv * np.matrix(self.d).T
        self.m = np.array(m).reshape(len(m))

        # resolution matrix
        if verbose:
            print "Setting up {0} x {0} resolution matrix (R)".format(self.G.shape[1])
        self.R = self.Ginv * self.Cinv * self.G

        # ===========================================================
        # Estimating spatial resolution at each node of the grid,
        # Rradius.
        #
        # The ith row of the resolution matrix, R[i,:], contains the
        # resolution map associated point grid node nb i, that is,
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
        for i, Ri in enumerate(np.array(self.R)):
            lon0, lat0 = self.grid.xy(i)

            # setting spatial resolution = nan for nodes without path
            if self.density[i] == 0:
                self.Rradius[i] = np.nan
                continue

            # best-fitting cone at point (lon0, lat0)

            def cone_height(lonlat, z0, r0):
                """
                Function returning height of cone of radius r0
                centered on (lon0, lat0), at point (lon, lat)
                """
                lon, lat = lonlat
                r = np.sqrt((lon - lon0)**2 + (lat - lat0)**2)
                return np.where(r < r0, z0 * (1 - r / r0), 0.0)

            # fitting the above function to heights in array Ri
            popt, _ = curve_fit(f=cone_height,
                                xdata=self.grid.xy_nodes(),
                                ydata=Ri, p0=[1, 2])

            # converting radius of best-fitting cone from decimal degrees
            # to km, along meridian (conservative choice as it yields the
            # largest possible value)
            _, r0 = popt
            d2rad = np.pi / 180.0
            r0 *= d2rad * 6371.0

            # resolution cannot be less than 2 * inter-node distance
            rmin = 2 * d2rad * 6371.0 * max(self.grid.xstep * np.cos(lat0 * d2rad),
                                            self.grid.ystep)
            r0 = max(rmin, r0)

            # appending spatial resolution to array
            self.Rradius[i] = r0

        # potting model
        self.plot()

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

    def plot(self, xsize=28):
        """
        Plots velocity perturbation, path density
        and spatial resolution
        """
        # bounding box
        bbox = self.grid.bbox()
        aspectratio = (bbox[3] - bbox[2]) / (bbox[1] - bbox[0])
        fig = plt.figure(figsize=(xsize, aspectratio*xsize/3.0), tight_layout=True)

        # plotting velocity perturbation
        ax = fig.add_subplot(131)
        self.plot_perturbation(ax)

        # plotting path density
        ax = fig.add_subplot(132)
        self.plot_pathdensity(ax)

        # plotting spatial resolution
        ax = fig.add_subplot(133)
        self.plot_resolution(ax)

        fig.show()

    def plot_pathdensity(self, ax=None, xsize=10, plotdensity=True, plotpaths=True,
                         stationlabel=False, showgrid=False):
        """
        Plots path density and/or interstation paths
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
            # plotting paths
            for path in self.paths:
                x, y = zip(*path)
                ax.plot(x, y, '-', color='grey', lw=0.5)

        if showgrid:
            # plotting grid
            x, y = self.grid.xy_nodes()
            ax.plot(x, y, '+')

        # plotting stations
        self._plot_stations(ax, stationlabel=stationlabel)

        # formatting axes
        ax.set_title(u'Period = {} s, {} paths'.format(self.period, len(self.paths)))
        ax.set_xlim(bbox[:2])
        ax.set_ylim(bbox[2:])

        if fig:
            fig.show()

    def plot_perturbation(self, ax=None, xsize=10):
        """
        Plots velocity perturbation in % relative to v0
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

        # plotting perturbation relative to reference velocity (%)
        # model params m = (v0 - v) / v0,  so perturbation = -m
        dv = -self.grid.to_2D_array(self.m)
        extent = (self.grid.xmin, self.grid.get_xmax(),
                  self.grid.ymin, self.grid.get_ymax())
        m = ax.imshow(100 * dv.transpose(), origin='bottom', extent=extent,
                      interpolation='bicubic', vmin=-10, vmax=10,
                      cmap=plt.get_cmap('seismic_r'))
        c = plt.colorbar(m, ax=ax, orientation='horizontal', pad=0.1)
        c.set_label('Velocity perturbation (%)')

        # formatting axes
        s = u'Period = {} s, {} paths'
        ax.set_title(s.format(self.period, len(self.paths)))
        ax.set_xlim(bbox[:2])
        ax.set_ylim(bbox[2:])

        if fig:
            fig.show()

    def plot_resolution(self, ax=None, xsize=10):
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
            fig = plt.figure(figsize=(xsize, aspectratio * xsize))
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
                      interpolation='bicubic', vmax=800,
                      cmap=plt.get_cmap('YlOrRd_r'))
        c = plt.colorbar(m, ax=ax, orientation='horizontal', pad=0.1)
        c.set_label('Spatial resolution (km)')

        # formatting axes
        s = u'Period = {} s, {} paths'
        ax.set_title(s.format(self.period, len(self.paths)))
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
    x1 = 1.0 / dmax
    x2 = 5.0 / dmax
    x3 = 10.0 / dmax
    cdict = {'red': ((0, 1, 1), (x1, 0, 0), (x2, 0, 0), (x3, 1, 1), (1, 0, 0)),
             'green': ((0, 1, 1), (x1, 0, 0), (x2, 1, 1), (x3, 0, 0), (1, 0, 0)),
             'blue': ((0, 1, 1), (x1, 1, 1), (x2, 0, 0), (x3, 0, 0), (1, 0, 0))}
    return LinearSegmentedColormap('tmp', cdict)

if __name__ == '__main__':
    # loading dispersion curves
    flist = sorted(glob.glob(pathname='../Cross-correlation/FTAN*.pickle*'))
    print 'Select file containing dispersion curves:'
    print '\n'.join('{} - {}'.format(i, os.path.basename(f)) for i, f in enumerate(flist))
    pickle_file = flist[int(raw_input('\n'))]
    f = open(pickle_file, 'rb')
    curves = pickle.load(f)
    f.close()
    print "Dispserion curves stored in variable 'curves'"