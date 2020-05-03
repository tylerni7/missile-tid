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
import connections


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
    'ncbx', 'ncdu',
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

station_data = {}
station_locs = {}
for station in stations:
    print(station)
    station_data[station] = defaultdict(lambda : defaultdict(lambda : None))
    date = start_date
    while date <= start_date + duration:
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
def vtec_for(station, prn, conns=None, biases=None):
    if biases:
        station_bias = biases.get(station, 0)
        sat_bias = biases.get(prn, 0)
    else:
        station_bias, sat_bias = 0, 0

    if prn not in station_vtecs[station]:
        dats = []
        locs = []
        slants = []
        if station_data[station][prn]:
            end = max(station_data[station][prn].keys())
        else:
            end = 0
        for i in range(end):
            measurement = station_data[station][prn][i]
            # if conns specified, require ambiguity data
            if conns:
                if measurement and conns[i] and conns[i].n1: # and numpy.std(conns[i].n1s) < 3:
                    res = tec.calc_vtec(
                        dog,
                        station_locs[station],
                        station_data[station][prn][i],
                        n1=conns[i].n1,
                        n2=conns[i].n2,
                        rcvr_bias=station_bias,
                        sat_bias=sat_bias,
                    )
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

            elif measurement:
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

def populate_data(conn_map=None):
    for station in stations:
        print(station)
        for recv in ['G%02d' % i for i in range(1, 33)]:
            if conn_map:
                vtec_for(station, recv, conns=conn_map[station][recv])
            else:
                vtec_for(station, recv)


# conn_map = solved_conn_map(dog, station_locs, station_data)
conns = connections.get_connections(dog, station_locs, station_data)
groups, unpaired = connections.get_groups(conns)
print(len(unpaired), "unpaired")
connections.correct_groups(station_locs, station_data, groups)
conn_map = connections.make_conn_map(conns)

populate_data(conn_map)

cal_dat = bias_solve.gather_data(station_vtecs)

import pickle
pickle.dump(cal_dat, open("cal_dat", "wb"))

def correct(vtecs, sat_biases, station_biases):
    for station in vtecs:
        for prn in ['G%02d' % i for i in range(1, 33)]:
            if prn not in vtecs[station]:
                continue
            for i in range(len(vtecs[station][prn][0])):
                dat = vtecs[station][prn][0][i], vtecs[station][prn][1][i], vtecs[station][prn][2][i]
                if dat[0] is None:
                    continue
                dat = tec.correct_tec(dat, rcvr_bias=station_biases[station], sat_bias=sat_biases[prn])
                vtecs[station][prn][0][i], vtecs[station][prn][1][i], vtecs[station][prn][2][i] = dat

def plot_station(vtecs, station):
    fig, ax = plt.subplots()
    for i in range(1, 33): 
        if 'G%02d' % i not in vtecs[station]:
            continue
        ax.plot([(x if x else math.nan) for x in vtecs[station]['G%02d' % i][1]])
    fig.show()


def plot_station_raw(vtecs, station):
    fig, ax = plt.subplots()
    for i in range(1, 33): 
        if 'G%02d' % i not in vtecs[station]:
            continue
        ax.plot([((x/y)*y**0.5 if x else math.nan) for x,y in zip(vtecs[station]['G%02d' % i][1], vtecs[station]['G%02d' % i][2])])
    fig.show()

def plot_station_depletion(vtecs, station):
    fig, ax = plt.subplots()
    for i in range(1, 33): 
        if 'G%02d' % i not in vtecs[station]:
            continue
        ax.plot(get_data.depletion_contiguous(
            [(x if x else math.nan) for x in vtecs[station]['G%02d' % i][1]],
            expected_len=2880*4
        ))
    fig.show()

def plot_station_filter(vtecs, station, short_min=2, long_min=12, ):
    fig, ax = plt.subplots()
    for i in range(1, 33): 
        if 'G%02d' % i not in vtecs[station]:
            continue
        ax.plot(get_data.filter_contiguous(
            [(x if x else math.nan) for x in vtecs[station]['G%02d' % i][1]],
            short_min=short_min, long_min=long_min, expected_len=2880*4
        ))
    fig.show()


def plot_map(vtec, stations):
    globe = Basemap(projection='mill',lon_0=180)
    # plot coastlines, draw label meridians and parallels.
    globe.drawcoastlines()
    #globe.drawparallels(numpy.arange(-90,90,30),labels=[1,0,0,0])
    #globe.drawmeridians(numpy.arange(globe.lonmin,globe.lonmax+30,60),labels=[0,0,0,1])

    scatter = globe.scatter([], [])

    def animate(i):
        plt.title( str(timedelta(seconds=i * 30) + start_date) )

        lons = []
        lats = []
        vals = []
        for station in stations:
            for prn in ['G%02d' % x for x in range(1, 33)]:
                if prn not in vtec[station] or i >= len(vtec[station][prn][0]):
                    continue
                try:
                    ecef = vtec[station][prn][0][i]
                except IndexError:
                    print(station, prn, i)
                    raise
                if ecef is None:
                    continue
                lat, lon, _ = coordinates.ecef2geodetic(ecef)
                lon = lon if lon > 0 else lon + 360
                lons.append(lon)
                lats.append(lat)

                vals.append(vtec[station][prn][1][i])

        scatter.set_offsets(numpy.array(globe(lons, lats)).T)
        max_tec = 60
        scatter.set_color( cm.plasma(numpy.array(vals) / max_tec) )

    def init():
        scatter.set_offsets([])

    ani = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=range(4000, 9000), repeat=True, interval=60)

    #globe.draw()
    #globe.show(block=False)
    plt.show()
    print("done")
    return ani

'''
gt = """   G01    -7.314     0.011                                  PRN / BIAS / RMS
   G02     7.514     0.011                                  PRN / BIAS / RMS
   G03    -5.238     0.011                                  PRN / BIAS / RMS
   G04    -0.985     0.011                                  PRN / BIAS / RMS
   G05     3.385     0.012                                  PRN / BIAS / RMS
   G06    -6.595     0.011                                  PRN / BIAS / RMS
   G07     3.311     0.011                                  PRN / BIAS / RMS
   G08    -7.209     0.011                                  PRN / BIAS / RMS
   G09    -4.670     0.011                                  PRN / BIAS / RMS
   G10    -5.533     0.011                                  PRN / BIAS / RMS
   G11     4.028     0.011                                  PRN / BIAS / RMS
   G12     3.988     0.011                                  PRN / BIAS / RMS
   G13     3.462     0.011                                  PRN / BIAS / RMS
   G14     2.297     0.011                                  PRN / BIAS / RMS
   G15     3.132     0.012                                  PRN / BIAS / RMS
   G16     2.982     0.011                                  PRN / BIAS / RMS
   G17     3.254     0.011                                  PRN / BIAS / RMS
   G18    -0.377     0.013                                  PRN / BIAS / RMS
   G19     6.136     0.011                                  PRN / BIAS / RMS
   G20     1.739     0.011                                  PRN / BIAS / RMS
   G21     2.752     0.011                                  PRN / BIAS / RMS
   G22     7.830     0.011                                  PRN / BIAS / RMS
   G23     9.256     0.011                                  PRN / BIAS / RMS
   G24    -5.732     0.011                                  PRN / BIAS / RMS
   G25    -7.704     0.011                                  PRN / BIAS / RMS
   G26    -8.456     0.011                                  PRN / BIAS / RMS
   G27    -5.065     0.011                                  PRN / BIAS / RMS
   G28     3.199     0.011                                  PRN / BIAS / RMS
   G29     2.682     0.011                                  PRN / BIAS / RMS
   G30    -6.459     0.011                                  PRN / BIAS / RMS
   G31     4.667     0.011                                  PRN / BIAS / RMS
   G32    -4.278     0.011                                  PRN / BIAS / RMS
"""
biases = {y[0]:float(y[1]) for y in [x.split() for x in gt.split("\n")][:-1]}

def print_coi(coi, res=None):
    for obs in sorted(coi, key=lambda x:(x.station, x.sat)):
        if res:
            print("\t".join([str(x) for x in (obs.station, obs.sat, obs.tec/obs.slant, (obs.tec/(obs.slant * 9.5186) + biases[obs.sat]), (obs.tec/(obs.slant) - res[obs.sat]) )]))
        else:
            print("\t".join([str(x) for x in (obs.station, obs.sat, obs.tec/obs.slant, (obs.tec/(obs.slant * 9.5186) + biases[obs.sat]) )]))

gcois = sorted(cal_dat[0], key=lambda x:len([o.station for o in cal_dat[0][x]]) - len({o.station for o in cal_dat[0][x]}))[-30:]

for coi in gcois:
    print(coi)
    print_coi(cal_dat[0][coi])
    print()

from collections import Counter

diffs = dict()
for i in range(32):
    diffs['G%02d' % i] = dict()

for c in cal_dat[0].values():
    cnt = Counter([o.station for o in c])
    for station, count in cnt.most_common():
        if count == 1:
            continue
        
        meases = [o for o in c if o.station == station]
        for m1, m2 in itertools.product(meases, repeat=2):
            if m1 == m2:
                continue
            if m2.sat not in diffs[m1.sat]:
                diffs[m1.sat][m2.sat] = []
            diffs[m1.sat][m2.sat].append( m2.tec/(m2.slant * 9.5186) -  m1.tec/(m1.slant * 9.5186) )
'''