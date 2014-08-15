__author__ = 'bruno'
from pysismo import pstomo
import sys
from pysismo.pscrosscorr import CrossCorrelationCollection, CrossCorrelation
from pysismo.psstation import Station
import glob
import pickle

# old module name = 'crosscorr'
sys.modules['pysismo.crosscorr'] = pstomo
#setattr(pstomo, 'Station', Station)

# looking for files containing pickled data
filelist = sorted(glob.glob(pathname='../Cross-correlation/xcorr*.pickle*'))
print "Found {} pickle files".format(len(filelist))

for filename in filelist:
    print "Processing file: " + filename
    f = open(name=filename, mode="rb")
    xc = pickle.load(f)
    f.close()

    # class should be pysismo.pscrosscorr.CrossCorrelationCollection
    #              or pysismo.pstomo.DispersionCurve
    print "  Object class: {}".format(xc.__class__)

    # # getting unique stations
    # stations = []
    # for s1, s2 in xc.pairs():
    #     for station in [xc[s1][s2].station1, xc[s1][s2].station2]:
    #         if station not in stations:
    #             stations.append(station)
    # print "  Converting {} stations".format(len(stations))
    #
    # # converting stations
    # newstations = []
    # for s in stations:
    #     newstation = Station(name=s.name,
    #                          network=s.network,
    #                          channel=s.channel,
    #                          filename=s.file,
    #                          basedir=s.basedir,
    #                          subdirs=s.subdirs,
    #                          coord=s.coord)
    #     newstations.append(newstation)
    #
    # # attaching new stations to cross-corr
    # for s1, s2 in xc.pairs():
    #     xc[s1][s2].station1 = next(s for s in newstations if s == xc[s1][s2].station1)
    #     xc[s1][s2].station2 = next(s for s in newstations if s == xc[s1][s2].station2)

    # dumping xc
    print "  Dumping to file: " + filename
    f = open(name=filename, mode="wb")
    pickle.dump(obj=xc, file=f, protocol=2)
    f.close()

    print