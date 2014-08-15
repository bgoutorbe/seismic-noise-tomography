#!/usr/bin/env python

# Script to get PAZ from GEOSCOPE station(s) since dataless seed from WebRequest
# is unreadable by obspy.xseed

from obspy.core import UTCDateTime
from obspy.arclink import Client

user = "brunog@id.uff.br"
network  = "G"
station  = "SPB"
location = ""
channel  = "BHZ"
starttime = UTCDateTime("2000-01-01")
endtime   = UTCDateTime("2000-02-01")

client = Client(user=user)
client.getPAZ(network, station, location, channel, starttime, endtime)
