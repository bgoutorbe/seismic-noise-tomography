import datetime as dt
import geopy
import geopy.distance
import geographiclib as gl
from geographiclib import geodesic
import pyproj

NLOOP = 1000

print 'Testing geopy'
t0 = dt.datetime.now()
for i in range(NLOOP):
    _ = geopy.distance.distance(geopy.Point(-10, 0), geopy.Point(-50, -50)).km
print (dt.datetime.now() - t0).total_seconds() * 1000

print 'Testing geographiclib'
t0 = dt.datetime.now()
for i in range(NLOOP):
    _ = geodesic.Geodesic.WGS84.Inverse(-10, 0, -50, -50, 
                                        outmask=gl.geodesic.Geodesic.ALL)
print (dt.datetime.now() - t0).total_seconds() * 1000

print 'Testing pyproj'
g = pyproj.Geod(ellps='WGS84')
t0 = dt.datetime.now()
for i in range(NLOOP):
    _ = g.inv(0, -10, -50, -50)
print (dt.datetime.now() - t0).total_seconds() * 1000

print 'Testing pyproj (with list)'
g = pyproj.Geod(ellps='WGS84')
t0 = dt.datetime.now()
_ = g.inv(NLOOP * [0], NLOOP * [-10], NLOOP * [-50], NLOOP * [-50])
print (dt.datetime.now() - t0).total_seconds() * 1000