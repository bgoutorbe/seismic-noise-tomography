#!/usr/bin/env python

from obspy.core import UTCDateTime
from obspy.iris import Client

network   = 'BL'
station   = 'BAMB'
location  = '*'
channel   = 'BHZ' 

# Date parameters
day = UTCDateTime(2002, 05, 15)

client = Client()
infos = client.availability(network, station, location, channel, day, day+60)

print '\nInformations on availability from client.availability:'
print infos

print 'Checking infos with client.getWaveform:'
st = client.getWaveform(network, station, location, channel, day, day+60)
print st
    