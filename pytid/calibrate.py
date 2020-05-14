from collections import defaultdict
from datetime import datetime, timedelta
import itertools
from laika import AstroDog, rinex_file
from laika.lib import coordinates
import math
import numpy
import os
import pickle
from scipy.signal import butter, lfilter, filtfilt, sosfiltfilt


from gnss import bias_solve, connections, get_data, tec


dog = AstroDog(cache_dir=os.environ['HOME'] + "/.gnss_cache/")
start_date = datetime(2020, 2, 15)
duration = timedelta(days=3)

# idk why but these stations gave weird results, DON'T USE THEM
bad_stations = [
    'nmsu'
]

stations = [
    'napl', 'bkvl', 'zefr', 'pbch', 'flwe', 'flbn',
    'flwe', 'ormd', 'dlnd', 'okcb', 'mmd1', 'bmpd', 'okte', 'blom',
    'utmn', 'nvlm', 'p345', 'slac', 'ndst', 'pamm', 'njmt', 'kybo', 'mtlw',
    'scsr', 'cofc', 'nmsu', 'azmp', 'wask', 'dunn', 'zjx1', 'talh', 'gaay',
    'ztl4', 'aldo', 'fmyr', 'crst', 'altu', 'mmd1', 'prjc', 'msin',
    'cola', 'alla', 'mspe', 'tn22', 'tn18', 'wvat', 'ines', 'freo', 'hnpt',
    'ncbx', 'ncdu', 'loyq', 'ict1', 'p143', 'mc09', 'neho', 'moca'
]

"""
# just florida for now...
stations = [
    'bkvl', 'crst', 'flbn', 'flwe',
    'fmyr', 'napl', 'okcb', 'ormd', 'pbch', 'pcla',
    'talh', 'zefr', 'zjx1', 'tn22',
]
"""

for station in bad_stations:
    if station in stations:
        stations.remove(station)

# get the basic data we need from rinex files
station_locs, station_data = get_data.populate_data(dog, start_date, duration, stations)

# get "connections", which are satellite to receiver periods in which
# the carrier phase cycles don't jump around
conns = []

# this will take some time if we don't already have it....
restored = []
for station in stations:
    connection_cache_name = "cached/connections_%s_%s_to_%s" % (
        station,
        start_date.strftime("%Y-%m-%d"),
        (start_date + duration).strftime("%Y-%m-%d")
    )
    if os.path.exists(connection_cache_name):
        restored.append(station)
        conns += pickle.load(open(connection_cache_name, "rb"))

# get connections without cycle slips
conns += connections.get_connections(dog, station_locs, station_data, skip=restored)
conn_map = connections.make_conn_map(conns)

# take cycle-slip free data, and solve integer ambiguities
# note: cache restored data will skip over this automatically
connections.correct_conns(station_locs, station_data, conns)

# that probably took a while! cache the connections...
for station in conn_map.keys():
    if station in restored:
        continue
    cache_name = "cached/connections_%s_%s_to_%s" % (
        station,
        start_date.strftime("%Y-%m-%d"),
        (start_date + duration).strftime("%Y-%m-%d")
    )
    station_conns = [conn for conn in conns if conn.station == station]
    pickle.dump(station_conns, open(cache_name, "wb"))

# get the vtecs with ambiguities handled
station_vtecs = get_data.get_vtec_data(dog, station_locs, station_data, conn_map=conn_map)

# split up the data into coincidences where multiple stations/sats are over
# the same section of ionosphere
cal_dat = bias_solve.gather_data(station_vtecs)

# use quadratic programming to estimate satellite and receiver differential code biases (DCBs)
errors, sat_biases, rcvr_biases, tecs, _ = bias_solve.opt_solve(*cal_dat)

# biases are good for ~days
pickle.dump(
    sat_biases,
    open("cached/satbiases_%s_to_%s" % (
        start_date.strftime("%Y-%m-%d"),
        (start_date + duration).strftime("%Y-%m-%d")
    ), "wb")
)
pickle.dump(
    rcvr_biases,
    open("cached/rcvrbiases_%s_to_%s" % (
        start_date.strftime("%Y-%m-%d"),
        (start_date + duration).strftime("%Y-%m-%d")
    ), "wb")
)

# use our shiny DCBs to update our vtec data
corrected_vtecs = get_data.correct_vtec_data(station_vtecs, sat_biases, rcvr_biases)

def compare_ion(true_tecs, start_date, tecs):
    """
    compare our ionosphere data to true data from CODE
    """
    errs = []
    for (lat, lon, ticks), our_tec in sorted(tecs.items()):
        date = start_date + timedelta(seconds=ticks*30)
        if (
            lat in true_tecs
            and lon in true_tecs[lat]
            and date in true_tecs[lat][lon]
        ):
            true_tec = true_tecs[lat][lon][date]
            print(lat, lon, date, "%5.2f %5.2f   %5.2f" % (our_tec, true_tec, abs(our_tec-true_tec)))
            errs.append(our_tec-true_tec)
    return errs