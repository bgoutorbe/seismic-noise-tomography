"""
Script to download stations inventories and export
them in xmlstation format
"""

from obspy.fdsn import Client
from obspy import UTCDateTime
import os

# output dir
OUTPUTDIR = u"../StationXML files"

# networks, start-end dates and limits
networks = ['BL', 'G', 'IU']
startdatetime = UTCDateTime(1988, 10, 1)
enddatetime = UTCDateTime(2012, 4, 1)
lonlim = (-75, -32)
latlim = (-35, 8)

# IRIS client
client = Client("IRIS")

# getting inventory
for network in networks:
    print "Downloading inventory of network: " + network

    # file name = e.g., IU.xml
    filename = network + ".xml"
    filepath = os.path.join(OUTPUTDIR, filename)
    inv = client.get_stations(starttime=startdatetime,
                              endtime=enddatetime,
                              network=network,
                              channel='BHZ',
                              level='response',
                              minlongitude=lonlim[0],
                              maxlongitude=lonlim[1],
                              minlatitude=latlim[0],
                              maxlatitude=latlim[1],
                              filename=filepath)