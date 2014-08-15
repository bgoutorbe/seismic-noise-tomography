#!/usr/bin/env python

from obspy.iris import Client
import obspy.core
from obspy.core import UTCDateTime
import os

# output dir
outfile = os.path.expanduser(r'~\Desktop\IRIS_DATA_FROM_OBSPY\1999-01\G.SPB.BHZ.mseed')

client = Client()
oneday = 24*3600

day = 1
st = obspy.core.Stream()
while day <= 31:
    print day,
    if day != 7:
        t = UTCDateTime(1999,1,day)
        sttmp = client.getWaveform('G', 'SPB', '*', 'BHZ', t, t+oneday)
        for tr in sttmp: st.append(tr)
    else:
        t1 = UTCDateTime('1999-01-07T00:00:00')
        t2 = UTCDateTime('1999-01-07T10:40:00')
        sttmp = client.getWaveform('G', 'SPB', '*', 'BHZ', t1, t2)
        for tr in sttmp: st.append(tr)
        t1 = UTCDateTime('1999-01-07T11:00:00')
        t2 = UTCDateTime('1999-01-08T00:00:00')
        sttmp = client.getWaveform('G', 'SPB', '*', 'BHZ', t1, t2)
        for tr in sttmp: st.append(tr)
    day += 1

st.write(outfile, 'MSEED')
