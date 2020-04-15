from collections import defaultdict
from datetime import datetime, timedelta
import itertools
from laika import AstroDog, rinex_file
from laika.lib import coordinates
import math
from matplotlib import animation
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy
import os
from scipy.signal import butter, lfilter, filtfilt, sosfiltfilt


import tec
import get_data
import bias_solve


dog = AstroDog(cache_dir=os.environ['HOME'] + "/.gnss_cache/")
start_date = datetime(2020, 2, 15)
duration = timedelta(days=3)

stations = [
    'napl', 'bkvl', 'zefr', 'pbch', 'flwe', 'ormd', 'flbn',
    'flwe', 'ormd', 'dlnd', 'okcb', 'mmd1', 'bmpd', 'okte', 'blom',
    'utmn', 'nvlm', 'p345', 'slac', 'ndst', 'pamm', 'njmt', 'kybo', 'mtlw',
    'scsr', 'cofc', 'nmsu', 'azmp', 'wask', 'dunn', 'zjx1', 'talh', 'gaay',
    'ztl4', 'aldo', 'fmyr', 'crst', 'altu', 'mmd1', 'prjc', 'msin',
    'cola', 'alla', 'mspe', 'tn22', 'tn18', 'wvat', 'ines', 'freo', 'hnpt',
    'ncbx', 'ncdu',
]


station_data = {}
station_locs = {}
for station in stations:
    print(station)
    station_data[station] = defaultdict(lambda : defaultdict(lambda : None))
    date = start_date
    while date < start_date + duration:
        try:
            loc, data = get_data.data_for_station(dog, station, date)
            station_data[station] = get_data.station_transform(
                                        data,
                                        start_dict=station_data[station],
                                        offset=int((date - start_date).total_seconds()/30)
                                    )
            station_locs[station] = loc
        except (ValueError, rinex_file.DownloadError):
            print("*** error with station " + station)
        date += timedelta(days=1)

station_vtecs = defaultdict(dict)
def vtec_for(station, prn):
    if prn not in station_vtecs[station]:
        dats = []
        locs = []
        slants = []
        for i in range(2880):
            measurement = station_data[station][prn][i]
            if measurement:
                res = tec.calc_vtec(dog, station_locs[station], station_data[station][prn][i])
                if res is None:
                    locs.append(None)
                    dats.append(math.nan)
                    slants.append(math.nan)
                else:
                    dats.append(res[0])
                    locs.append(res[1])
                    slants.append(res[2])
            else:
                locs.append(None)
                dats.append(math.nan)
                slants.append(math.nan)
        
        station_vtecs[station][prn] = (locs, dats, slants)
    return station_vtecs[station][prn]

def populate_data():
    for station in stations:
        for recv in ['G%02d' % i for i in range(1, 32)]:
            vtec_for(station, recv)

populate_data()

cal_dat = bias_solve.gather_data(station_vtecs)
