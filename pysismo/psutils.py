# -*- coding: utf-8 -*-
"""
General utilities
"""

import obspy.signal.filter
import numpy as np
import os
import shutil
import shapefile
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pyproj

# locations to skip when cleaning stream
SKIPLOCS = ('50',)

# Map parameters
SHP_BASEDIR = '../shapefiles'
COAST_SHP = os.path.join(SHP_BASEDIR, 'SouthAmericaCoasts.shp')
TECTO_SHP = os.path.join(SHP_BASEDIR, 'SouthAmericaTectonicElements.shp')
TECTO_LABELS = os.path.join(SHP_BASEDIR, 'SouthAmericaTectonicElementsLabels.shp')
TECTO_COLORS = {
    'Archean': (0.98, 0.88, 1, 1),
    'Phanerozoic': (1, 0.988, 0.831, 1),
    'Neoproterozoic': '0.9'
}

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

    if len(set([tr.id for tr in st])) > 1:
        raise Exception('More than one trace! {0}'.format(set(tr.id for tr in st)))

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


def clean_stream(stream, skiplocs=SKIPLOCS, verbose=False):
    """
    1 - Removes traces whose location is in skiplocs.
    2 - Select trace from 1st location if several ids.

    @type stream: L{obspy.core.Stream}
    @type skiplocs: tuple of str
    @rtype: None
    """

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


def bandpass(data, df, tmin, tmax, corners=2, zerophase=True):
    """
    Bandpassing data of array *data* between tmin-tmax

    @type data: L{numpy.ndarray}
    @type df: float
    @type tmin: float or int or None
    @type tmax: float or int or None
    @type corners: int
    @type zerophase: bool
    @rtype: L{numpy.ndarray}
    """
    return obspy.signal.filter.bandpass(data=data, freqmin=1.0 / tmax,
                                        freqmax=1.0 / tmin, df=df,
                                        corners=corners, zerophase=zerophase)


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
    Plot bas map: coasts and tectonic provinces
    """
    fig = None
    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # plotting coasts
    sf = shapefile.Reader(COAST_SHP)
    for shape in sf.shapes():
        # adding polygon(s)
        parts = list(shape.parts) + [len(shape.points)]
        partlims = zip(parts[:-1], parts[1:])
        for i1, i2 in partlims:
            points = shape.points[i1:i2]
            x, y = zip(*points)
            ax.plot(x, y, '-', lw=1, color='k')

    # plotting tectonic provinces
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
                                           edgecolor='black', linewidths=0.5)
            ax.add_collection(tectprovince)
        else:
            for i1, i2 in partlims:
                x, y = zip(*shape.points[i1:i2])
                ax.plot(x, y, '-', color='gray', lw=0.5)

    if labels:
        # plotting tectonic labels withint bounding box
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