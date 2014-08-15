#!/usr/bin/env python

from obspy.core import UTCDateTime
from obspy.iris import Client

outfile="Stations.xml"
channel="BHZ"
starttime="1988-01-01"
endtime="2012-02-21"

client = Client()

t1 = UTCDateTime(starttime)
t2 = UTCDateTime(endtime)

print "Downloading available stations from " + starttime + " to " + endtime
print "with channel: "  + channel
stations = client.station("*", "*", channel=channel, starttime = t1, endtime = t2)

print "Writing xml stations to file " + outfile
file = open(outfile, mode="w")
file.write(stations)
file.close()