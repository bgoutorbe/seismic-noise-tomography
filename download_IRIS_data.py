#!/usr/bin/env python

"""
WARNING: this script is obsolete, since obspy.iris client
does not work anymore. obspy.fdsn client should be used
instead.
"""

from obspy.iris import Client
import obspy.core
from obspy.core import UTCDateTime
import numpy as np
import os
from time import sleep

def mseed_name(*args):
    name = [x for x in args if x != '*']
    return '.'.join(name) + '.mseed'

# time interval wihtin which downloading data
firstyear  = 2003
firstmonth = 10
lastyear   = 2012
lastmonth  = 3
oneday     = 24*3600

# networks/stations/locations/channels from which downloading data
download = dict()
#download['BL'] = {'stations':['*']                 , 'loc':'*' , 'channel':'BHZ'}
#download['G']  = {'stations':['SPB']               , 'loc':'*' , 'channel':'BHZ'}
download['IU'] = {'stations':['PTGA','RCBR','SAML'], 'loc':'*', 'channel':'BHZ'}

# skip day if max trace value > value below
maxval = 1.0e9

# output dir
outdir = os.path.expanduser(r'~\Desktop\IRIS_DATA_FROM_OBSPY')

client = Client()

year  = firstyear
month = firstmonth
while UTCDateTime(year,month,01) <= UTCDateTime(lastyear,lastmonth,01):
    starttime = UTCDateTime(year,month,01)
    endtime   = UTCDateTime(year,month+1,01) if month<12 else UTCDateTime(year+1,01,01)
    print '\n' + 10*'*' + ' Downloading data in month ' + starttime.strftime('%m-%Y') + ' ' + 10*'*'
    
    for network in download:
        #if network=='BL' and year==2001 and month==12: continue
        print '\nNetwork ' + network,
        # checking availability of all stations
        av = client.availability(network, '*', download[network]['loc'], download[network]['channel'],
                                 starttime, endtime)
        av = av.split('\n')
        
        # set of available stations for this year-month
        station_set = set(x.split(' ')[1] for x in av if len(x.strip()) > 0)

        # intersection with desired stations
        if '*' not in download[network]['stations']:
            station_set = station_set.intersection(set(download[network]['stations']))
            
        print '-> ' + str(len(station_set)) + ' stations available'
        
        # downloading data of stations in set of available stations
        for station in np.sort([x for x in station_set]):
            #if (station=='BAMB' or station=='BEB' or station=='CAUB' or station=='CORB') and year==2003 and month==3: continue
            print '- ' + station,
                
            st = obspy.core.Stream()
#            try:
#                st = client.getWaveform(network, station,download[network]['loc'], \
#                                        download[network]['channel'],starttime,endtime)
#            except Exception as inst:
#                st = obspy.core.Stream()
#                if 'No waveform data available' in inst.message:
#                    print 'no data',
#                else:
#                    # trying to download data day by day
#                    print type(inst), '- trying day by day',
            t = starttime
            while t < endtime:
                print t.date.day,
                while True:
                    try:
                        # downloading traces of the day and appending to stream
                        sttmp = client.getWaveform(network, station, download[network]['loc'],
                                                   download[network]['channel'],t,t+oneday)
                        for tr in sttmp:
                            max = np.max(abs(tr.data))                                 
                            if max < maxval:                                    
                                st.append(tr)
                            else:
                                print 'bad data - skipping trace',
                        break
                    except Exception as inst:
                        if 'No waveform data available' in inst.message:
                            print 'no data',
                            break
                        else:
                            print type(inst), '- trying again',
                            sleep(10)
                t += oneday
            print

            # writing waveform as miniseed
            if len(st) > 0:
                basename = mseed_name(network, station,download[network]['channel'],
                                      download[network]['loc'])
                dirpath  = os.path.join(outdir, str(year) + '-' + "%02d" % month)            
                outfile  = os.path.join(dirpath, basename)
                if not os.path.isdir(dirpath): os.makedirs(dirpath)
                st.write(outfile, 'MSEED')
        print
    
    year  = year+1 if month==12 else year
    month = month+1 if month<12 else 1
