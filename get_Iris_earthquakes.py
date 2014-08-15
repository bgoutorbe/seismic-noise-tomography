#!/usr/bin/env python

from obspy.core import UTCDateTime
from obspy.iris import Client

outfile="Earthquakes.xml"
minmag=5.5
starttime="1988-01-01"
endtime="2012-02-21"

client = Client()

t1 = UTCDateTime(starttime)
t2 = UTCDateTime(endtime)

print "Downloading events of magnitude >",minmag, "from " + starttime + " to " + endtime
events = client.event(minmag = minmag, starttime = starttime, endtime = endtime)

print "Writing xml events to file " + outfile
file = open(outfile, mode="w")
file.write(events)
file.close()