# -*- coding: utf-8 -*-
"""
General utilities
"""

import obspy.signal.filter
import numpy as np
from numpy.lib.stride_tricks import as_strided
from numpy.fft import rfft, irfft, rfftfreq
import os
import glob
import shutil
import pickle
import shapefile
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pyproj
import itertools as it
from pyPdf import PdfFileReader, PdfFileWriter

# ====================================================
# parsing configuration file to import some parameters
# ====================================================
from psconfig import CROSSCORR_SKIPLOCS, COAST_SHP, TECTO_SHP, TECTO_LABELS, TECTO_COLORS

# reference elipsoid to calculate distance
wgs84 = pyproj.Geod(ellps='WGS84')


def filelist(basedir, ext=None, subdirs=True):
    """
    Returns the list of files in *basedir* (and subdirs if
    *subdirs* is True) whose extendion is *ext*
    """
    # list of files and dirs
    flist = os.listdir(basedir)
    files = []
    for f in flist:
        if os.path.isfile(os.path.join(basedir, f)):
            if not ext:
                files.append(f)
            elif os.path.splitext(f)[1].lower() == "." + ext.lower():
                files.append(f)
    if subdirs:
        for d in flist:
            if os.path.isdir(os.path.join(basedir, d)):
                subdir = os.path.join(basedir, d)
                sublist = filelist(subdir, ext=ext, subdirs=True)
                for f in sublist:
                    files.append(os.path.join(d, f))
    return files


def openandbackup(filename, mode='w'):
    """
    Opens file, backing up older version if file exists.

    @type filename: str or unicode
    @rtype: file
    """
    if os.path.exists(filename):
        # backup
        shutil.copyfile(filename, filename + '~')
        # opening file
    f = open(filename, mode=mode)
    return f


def get_fill(st, starttime=None, endtime=None):
    """
    Subroutine to get data fill
    @rtype: float
    """
    if len(st) == 0:
        # no trace
        return 0.0

    ststart = min(tr.stats.starttime for tr in st)
    stend = max(tr.stats.endtime for tr in st)
    dttot = (stend if not endtime else endtime) - \
            (ststart if not starttime else starttime)
    gaps = st.getGaps()

    fill = 1.0
    if starttime:
        fill -= max(ststart - starttime, 0.0) / dttot
    if endtime:
        fill -= max(endtime - stend, 0.0) / dttot

    for g in gaps:
        gapstart = g[4]
        gapend = g[5]
        if starttime:
            gapstart = max(gapstart, starttime)
            gapend = max(gapend, starttime)
        if endtime:
            gapstart = min(gapstart, endtime)
            gapend = min(gapend, endtime)
        fill -= (gapend - gapstart) / dttot

    return fill


def clean_stream(stream, skiplocs=CROSSCORR_SKIPLOCS, verbose=False):
    """
    1 - Removes traces whose location is in skiplocs.
    2 - Select trace from 1st location if several ids.

    @type stream: L{obspy.core.Stream}
    @type skiplocs: tuple of str
    @rtype: None
    """

    if not skiplocs:
        skiplocs = []

    # removing traces of stream from locations to skip
    for tr in [tr for tr in stream if tr.stats.location in skiplocs]:
        stream.remove(tr)

    # if more than one id -> taking first location (sorted alphanumerically)
    if len(set(tr.id for tr in stream)) > 1:
        locs = sorted(set(tr.stats.location for tr in stream))
        select_loc = locs[0]
        if verbose:
            s = "warning: selecting loc '{loc}', discarding locs {locs}"
            print s.format(loc=select_loc, locs=','.join(locs[1:])),
        for tr in [tr for tr in stream if tr.stats.location != select_loc]:
            stream.remove(tr)


def plot_nb_pairs():
    """
    Plot the total nb of group velocity measurements and the remaining
    nb of measurements (after applying selection criteria), function of
    period, for the selected dispersion curves.
    """
    # parsing some parameters of configuration file
    from pysismo.psconfig import (FTAN_DIR, MINSPECTSNR, MINSPECTSNR_NOSDEV,
                                  MINNBTRIMESTER, MAXSDEV)

    # selecting dispersion curves
    flist = sorted(glob.glob(os.path.join(FTAN_DIR, 'FTAN*.pickle*')))
    print 'Select file(s) containing dispersion curves to process:'
    print '\n'.join('{} - {}'.format(i, os.path.basename(f))
                    for i, f in enumerate(flist))
    res = raw_input('\n')
    pickle_files = [flist[int(i)] for i in res.split()]

    for curves_file in pickle_files:
        # loading dispersion curves of file
        print "Loading file: " + curves_file
        f = open(curves_file, 'rb')
        curves = pickle.load(f)
        f.close()
        periods = curves[0].periods

        # updating selection parameters of dispersion curves
        for c in curves:
            c.update_parameters(minspectSNR=MINSPECTSNR,
                                minspectSNR_nosdev=MINSPECTSNR_NOSDEV,
                                minnbtrimester=MINNBTRIMESTER,
                                maxsdev=MAXSDEV)

        # list of arrays of filtered velocities
        list_filtered_vels = [c.filtered_vels_sdevs()[0] for c in curves]

        n_init = []
        n_final = []

        for period in periods:
            iperiods = [c.get_period_index(period) for c in curves]

            # total nb of mesurements
            vels = np.array([c.v[i] for c, i in zip(curves, iperiods)])
            n_init.append(np.count_nonzero(~np.isnan(vels)))

            # remaining nb of measurements after selection criteria
            vels = np.array([v[i] for v, i in zip(list_filtered_vels, iperiods)])
            n_final.append(np.count_nonzero(~np.isnan(vels)))

        lines = plt.plot(periods, n_init, label=os.path.basename(curves_file))
        plt.plot(periods, n_final, color=lines[0].get_color())

    # finalizing and showing plot
    plt.xlabel('Period (s)')
    plt.ylabel('Nb of measurements')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True)
    plt.show()

def resample(trace, dt_resample):
    """
    Subroutine to resample trace

    @type trace: L{obspy.core.trace.Trace}
    @type dt_resample: float
    @rtype: L{obspy.core.trace.Trace}
    """
    dt = 1.0 / trace.stats.sampling_rate
    factor = dt_resample / dt
    if int(factor) == factor:
        # simple decimation (no filt because it shifts the data)
        trace.decimate(int(factor), no_filter=True)
    else:
        # linear interpolation
        tp = np.arange(0, trace.stats.npts) * trace.stats.delta
        zp = trace.data
        ninterp = int(max(tp) / dt_resample) + 1
        tinterp = np.arange(0, ninterp) * dt_resample

        trace.data = np.interp(tinterp, tp, zp)
        trace.stats.npts = ninterp
        trace.stats.delta = dt_resample
        trace.stats.sampling_rate = 1.0 / dt_resample
        #trace.stats.endtime = trace.stats.endtime + max(tinterp)-max(tp)


def moving_avg(a, halfwindow, mask=None):
    """
    Performs a fast n-point moving average of (the last
    dimension of) array *a*, by using stride tricks to roll
    a window on *a*.

    Note that *halfwindow* gives the nb of points on each side,
    so that n = 2*halfwindow + 1.

    If *mask* is provided, values of *a* where mask = False are
    skipped.

    Returns an array of same size as *a* (which means that near
    the edges, the averaging window is actually < *npt*).
    """
    # padding array with zeros on the left and on the right:
    # e.g., if halfwindow = 2:
    # a_padded    = [0 0 a0 a1 ... aN 0 0]
    # mask_padded = [F F ?  ?      ?  F F]

    if mask is None:
        mask = np.ones_like(a, dtype='bool')

    zeros = np.zeros(a.shape[:-1] + (halfwindow,))
    falses = zeros.astype('bool')

    a_padded = np.concatenate((zeros, np.where(mask, a, 0), zeros), axis=-1)
    mask_padded = np.concatenate((falses, mask, falses), axis=-1)

    # rolling window on padded array using stride trick
    #
    # E.g., if halfwindow=2:
    # rolling_a[:, 0] = [0   0 a0 a1 ...    aN]
    # rolling_a[:, 1] = [0  a0 a1 a2 ... aN 0 ]
    # ...
    # rolling_a[:, 4] = [a2 a3 ...    aN  0  0]

    npt = 2 * halfwindow + 1  # total size of the averaging window
    rolling_a = as_strided(a_padded,
                           shape=a.shape + (npt,),
                           strides=a_padded.strides + (a.strides[-1],))
    rolling_mask = as_strided(mask_padded,
                              shape=mask.shape + (npt,),
                              strides=mask_padded.strides + (mask.strides[-1],))

    # moving average
    n = rolling_mask.sum(axis=-1)
    return np.where(n > 0, rolling_a.sum(axis=-1).astype('float') / n, np.nan)


def local_maxima_indices(x, include_edges=True):
    """
    Returns the indices of all local maxima of an array x
    (larger maxima first)

    @type x: L{numpy.ndarray}
    @rtype: list of int
    """
    mask = (x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:])
    indices = np.nonzero(mask)[0] + 1
    if include_edges:
        # local maxima on edges?
        if x[0] >= x[1]:
            indices = np.r_[0, indices]
        if x[-1] >= x[-2]:
            indices = np.r_[len(x) - 1, indices]
    indices = sorted(indices, key=lambda index: x[index], reverse=True)
    return indices


def bandpass(data, dt, filtertype='Butterworth', **kwargs):
    """
    Bandpassing array *data* (whose sampling step is *dt*)
    using either a Butterworth filter (filtertype='Butterworth')
    or a Gaussian filter (filtertype='Gaussian')

    Additional arguments in *kwargs* are sent to
    bandpass_butterworth() (arguments: periodmin, periodmax,
    corners, zerophase) or bandpass_gaussian() (arguments:
    period, alpha)

    @type data: L{numpy.ndarray}
    @type dt: float
    @rtype: L{numpy.ndarray}
    """
    if filtertype.lower().strip() == 'butterworth':
        return bandpass_butterworth(data, dt, **kwargs)
    elif filtertype.lower().strip() == 'gaussian':
        return bandpass_gaussian(data, dt, **kwargs)
    else:
        raise Exception("Unknown filter: " + filtertype)


def bandpass_butterworth(data, dt, periodmin, periodmax, corners=2, zerophase=True):
    """
    Bandpassing data (in array *data*) between periods
    *periodmin* and *periodmax* with a Butterworth filter.
    *dt* is the sampling interval of the data.

    @type data: L{numpy.ndarray}
    @type dt: float
    @type periodmin: float or int or None
    @type periodmax: float or int or None
    @type corners: int
    @type zerophase: bool
    @rtype: L{numpy.ndarray}
    """
    return obspy.signal.filter.bandpass(data=data, freqmin=1.0 / periodmax,
                                        freqmax=1.0 / periodmin, df=1.0 / dt,
                                        corners=corners, zerophase=zerophase)


def bandpass_gaussian(data, dt, period, alpha):
    """
    Bandpassing real data (in array *data*) with a Gaussian
    filter centered at *period* whose width is controlled
    by *alpha*:

      exp[-alpha * ((f-f0)/f0)**2],

    with f the frequency and f0 = 1 / *period*.
    *dt* is the sampling interval of the data.

    @type data: L{numpy.ndarray}
    @type dt: float
    @type period: float
    @type alpha: float
    @rtype: L{numpy.ndarray}
    """
    # Fourier transform
    fft_data = rfft(data)
    # aray of frequencies
    freq = rfftfreq(len(data), d=dt)

    # bandpassing data
    f0 = 1.0 / period
    fft_data *= np.exp(-alpha * ((freq - f0) / f0) ** 2)

    # back to time domain
    return irfft(fft_data, n=len(data))


def dist(lons1, lats1, lons2, lats2):
    """
    Returns an array of geodetic distance(s) in km between
    points (lon1, lat1) and (lon2, lat2)
    """
    _, _, d = wgs84.inv(lons1=lons1, lats1=lats1, lons2=lons2, lats2=lats2)
    return np.array(d) / 1000.0


def geodesic(coord1, coord2, npts):
    """
    Returns a list of *npts* points along the geodesic between
    (and including) *coord1* and *coord2*, in an array of
    shape (*npts*, 2).
    @rtype: L{ndarray}
    """
    if npts < 2:
        raise Exception('nb of points must be at least 2')

    path = wgs84.npts(lon1=coord1[0], lat1=coord1[1],
                      lon2=coord2[0], lat2=coord2[1],
                      npts=npts-2)
    return np.array([coord1] + path + [coord2])


def geo2cartesian(lons, lats, r=1.0):
    """
    Converts geographic coordinates to cartesian coordinates
    """
    # spherical coordinates
    phi = np.array(lons) * np.pi / 180.0
    theta = np.pi / 2.0 - np.array(lats) * np.pi / 180.0
    # cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z


def projection(M, A, B, C):
    """
    Orthogonal projection of point(s) M on plane(s) ABC.
    Each point (M, A, B, C) should be a tuple of floats or
    a tuple of arrays, (x, y, z)
    """
    AB = vector(A, B)
    AC = vector(A, C)
    MA = vector(M, A)

    # unit vector u perpendicular to ABC (u = AB x AC / |AB x AC|)
    u = vectorial_product(AB, AC)
    norm_u = norm(u)
    u = [u[i] / norm_u for i in (0, 1, 2)]

    # (MA.u)u = MM' (with M' the projection of M on the plane)
    MA_dot_u = sum(MA[i] * u[i] for i in (0, 1, 2))
    MMp = [MA_dot_u * u[i] for i in (0, 1, 2)]
    xMp, yMp, zMp = [MMp[i] + M[i] for i in (0, 1, 2)]

    return xMp, yMp, zMp


def barycentric_coords(M, A, B, C):
    """
    Barycentric coordinates of point(s) M in triangle(s) ABC.
    Each point (M, A, B, C) should be a tuple of floats or
    a tuple of arrays, (x, y, z).
    Barycentric coordinate wrt A (resp. B, C) is the relative
    area of triangle MBC (resp. MAC, MAB).
    """
    MA = vector(M, A)
    MB = vector(M, B)
    MC = vector(M, C)

    # area of triangle = norm of vectorial product / 2
    wA = norm(vectorial_product(MB, MC)) / 2.0
    wB = norm(vectorial_product(MA, MC)) / 2.0
    wC = norm(vectorial_product(MA, MB)) / 2.0
    wtot = wA + wB + wC

    return wA / wtot, wB / wtot, wC / wtot


def vector(A, B):
    """
    Vector(s) AB. A and B should be tuple of floats or
    tuple of arrays, (x, y, z).
    """
    return tuple(np.array(B[i]) - np.array(A[i]) for i in (0, 1, 2))


def vectorial_product(u, v):
    """
    Vectorial product u x v. Vectors u, v should be tuple of
    floats or tuple of arrays, (ux, uy, uz) and (vx, vy, vz)
    """
    return (u[1]*v[2] - u[2]*v[1],
            u[2]*v[0] - u[0]*v[2],
            u[0]*v[1] - u[1]*v[0])


def norm(u):
    """
    Norm of vector(s) u, which should be a tuple of
    floats or a tuple of arrays, (ux, uy, uz).
    """
    return np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)


def basemap(ax=None, labels=True, axeslabels=True, fill=True, bbox=None):
    """
    Plots base map: coasts (file *COAST_SHP*), tectonic provinces
    file  *TECTO_SHP*) and labels (file *TECTO_LABELS*). Labels are
    plotted if *labels* = True. Tectonic provinces are filled
    (according to colors in dict *TECTO_COLORS*) if *fill* = True.
    """
    fig = None
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # plotting coasts
    if COAST_SHP:
        sf = shapefile.Reader(COAST_SHP)
        for shape in sf.shapes():
            # adding polygon(s)
            parts = list(shape.parts) + [len(shape.points)]
            partlims = zip(parts[:-1], parts[1:])
            for i1, i2 in partlims:
                points = shape.points[i1:i2]
                x, y = zip(*points)
                ax.plot(x, y, '-', lw=0.75, color='k')

    # plotting tectonic provinces
    if TECTO_SHP:
        sf = shapefile.Reader(TECTO_SHP)
        for sr in sf.shapeRecords():
            tectcategory = sr.record[0]
            color = next((TECTO_COLORS[k] for k in TECTO_COLORS.keys()
                         if k in tectcategory), 'white')
            shape = sr.shape
            parts = list(shape.parts) + [len(shape.points)]
            partlims = zip(parts[:-1], parts[1:])
            if fill:
                polygons = [Polygon(shape.points[i1:i2]) for i1, i2 in partlims]
                tectprovince = PatchCollection(polygons, facecolor=color,
                                               edgecolor='0.663', linewidths=0.5)
                ax.add_collection(tectprovince)
            else:
                for i1, i2 in partlims:
                    x, y = zip(*shape.points[i1:i2])
                    ax.plot(x, y, '-', color='0.663', lw=0.5)

    if labels and TECTO_LABELS:
        # plotting tectonic labels within bounding box
        sf = shapefile.Reader(TECTO_LABELS)
        for sr in sf.shapeRecords():
            label, angle = sr.record
            label = label.replace('\\', '\n')
            label = label.replace('Guapore', u'Guaporé').replace('Sao', u'São')
            x, y = sr.shape.points[0]
            if not bbox or bbox[0] < x < bbox[1] and bbox[2] < y < bbox[3]:
                ax.text(x, y, label, ha='center', va='center', color='grey',
                        fontsize=10, weight='bold', rotation=angle)

    # setting up axes
    ax.set_aspect('equal')
    if axeslabels:
        ax.set_xlabel('longitude (deg)')
        ax.set_ylabel('latitude (deg)')
        ax.grid(True)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(False)
    if bbox:
        ax.set_xlim(bbox[:2])
        ax.set_ylim(bbox[2:])

    if fig:
        fig.show()


def combine_pdf_pages(pdfpath, pagesgroups, verbose=False):
    """
    Combines vertically groups of pages of a pdf file

    @type pdfpath: str or unicode
    @type pagesgroups: list of (list of int)
    """
    # opening input file
    if verbose:
        print "Opening file " + pdfpath
    fi = open(pdfpath, 'rb')
    pdf = PdfFileReader(fi)

    # opening output pdf
    pdfout = PdfFileWriter()

    # loop on groups of pages tom combine
    for pagesgroup in pagesgroups:
        if verbose:
            print "Combining pages:",

        # heights and widths
        heights = [pdf.pages[i].mediaBox.getHeight() for i in pagesgroup]
        widths = [pdf.pages[i].mediaBox.getWidth() for i in pagesgroup]

        # adding new blank page
        page_out = pdfout.addBlankPage(width=max(widths), height=sum(heights))
        # merging pages of group
        for i, p in enumerate(pagesgroup):
            if verbose:
                print p,
            page_out.mergeTranslatedPage(pdf.pages[p], tx=0, ty=sum(heights[i+1:]))
        print

    # exporting merged pdf into temporary output file
    fo = create_tmpfile('wb')
    if verbose:
        print "Exporting merged pdf in file {}".format(fo.name)
    pdfout.write(fo)

    # closing files
    fi.close()
    fo.close()

    # removing original file and replacing it with merged pdf
    if verbose:
        print "Moving exported pdf to: " + pdfpath
    os.remove(pdfpath)
    os.rename(fo.name, pdfpath)


def create_tmpfile(*args, **kwargs):
    """
    Creates, opens and returns the first file tmp<i> that does
    not exist (with i = integer).
    *args and **kwargs are sent to open() function
    """
    for i in it.count():
        filepath = 'tmp{}'.format(i)
        if not os.path.exists(filepath):
            f = open(filepath, *args, **kwargs)
            return f


def groupbykey(iterable, key=None):
    """
    Returns a list of sublists of *iterable* grouped by key:
    all elements x of a given sublist have the same
    value key(x).

    key(x) must return a hashable object, such that
    set(key(x) for x in iterable) is possible.

    If not given, key() is the identity funcion.
    """
    if not key:
        key = lambda x: x

    # unique keys
    iterable = list(iterable)
    keys = set(key(x) for x in iterable)

    groups = []
    for k in keys:
        # group with key = k
        groups.append([x for x in iterable if key(x) == k])

    return groups
