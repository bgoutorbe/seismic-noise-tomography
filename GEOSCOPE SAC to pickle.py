#!/usr/bin/env python

# Dump metadata of station G.SPB in the form of a dict
# into a pickle file, because obspy.xseed is not able to
# read dataless seed file

from obspy.core import Trace, UTCDateTime
from obspy.sac import attach_paz
import pickle

outfile = r'../Dataless seed/G.SPB.pickle'

channeldicts = []
tr = Trace()

attach_paz(tr,r'../Dataless seed/GEOSCOPE/G.SPB..1996-2004.sac')
tr.stats.paz.sensitivity = 2.44531E+08
channeldict = dict(channelid='G.SPB..BHZ',
                   startdate=UTCDateTime('1996-06-17T00:00:00'),
                   enddate=UTCDateTime('2004-10-17T22:45:00'),
                   paz=tr.stats.paz)
channeldicts.append(channeldict)

attach_paz(tr,r'../Dataless seed/GEOSCOPE/G.SPB.10.2006-2011.sac')
tr.stats.paz.sensitivity = 4.006400e+09
channeldict = dict(channelid='G.SPB.10.BHZ',
                   startdate=UTCDateTime('2006-11-02T11:12:00'),
                   enddate=UTCDateTime('2011-12-02T11:51:00'),
                   paz=tr.stats.paz)
channeldicts.append(channeldict)


attach_paz(tr,r'../Dataless seed/GEOSCOPE/G.SPB.10.2011-2012.sac')
tr.stats.paz.sensitivity = 5.789880e+09
channeldict = dict(channelid='G.SPB.10.BHZ',
                   startdate=UTCDateTime('2011-12-10T00:00:00'),
                   enddate=None,
                   paz=tr.stats.paz)
channeldicts.append(channeldict)

print channeldicts

f = open(outfile, 'wb')
pickle.dump(channeldicts, f, protocol=2)
f.close()

print 'Metadata dumped to file {f} for the above station'.format(f=outfile)