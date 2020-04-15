from collections import defaultdict
from datetime import datetime
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


dog = AstroDog(cache_dir=os.environ['HOME'] + "/.gnss_cache/")

date = datetime(2020, 2, 17)
#date = datetime(2020, 2, 16)

print("loading data...")
stations = [
    'napl', 'bkvl', 'zefr', 'pbch', 'flwe', 'ormd', 'flbn',
    'flwe', 'ormd', 'dlnd', 'okcb', 'mmd1', 'bmpd', 'okte', 'blom',
    'utmn', 'nvlm', 'p345', 'slac', 'ndst', 'pamm', 'njmt', 'kybo', 'mtlw',
    'scsr', 'cofc', 'nmsu', 'azmp', 'wask', 'dunn', 'zjx1', 'talh', 'gaay',
    'ztl4', 'aldo', 'fmyr', 'crst', 'altu', 'mmd1', 'prjc', 'msin',
    'cola', 'alla', 'mspe', 'tn22', 'tn18', 'wvat', 'ines', 'freo', 'hnpt',
    'ncbx', 'ncdu',
]
# just florida stuff
"""
stations = [
    'bkvl', 'brtw', 'crst', 'flbn', 'flf1', 'flwe',
    'fmyr', 'gnvl', 'laud', 'mtnt', 'napl', 'okcb', 'ormd', 'pbch', 'pcla',
    'pltk', 'prry', 'talh', 'xcty', 'zefr', 'zjx1', 'zma1',
]
"""
#stations = ['napl', 'flwe', 'ormd', 'gaay', 'ncdu', 'tn22', 'flbn']


station_data = {}
station_locs = {}
for station in stations:
    try:
        loc, data = get_data.data_for_station(dog, station, date)
        station_data[station] = get_data.station_transform(data)
        station_locs[station] = loc
    except (ValueError, rinex_file.DownloadError):
        print("*** error with station " + station)

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

def get_bpfilt_for(station, prn, short_min=2, long_min=12):
    locs, dats, _ = vtec_for(station, prn)
    get_data.remove_slips(dats)
    return locs, (get_data.filter_contiguous(dats, short_min=short_min, long_min=long_min))

def get_depletion_for(station, prn):
    locs, dats, _ = vtec_for(station, prn)
    get_data.remove_slips(dats)
    return locs, get_data.depletion_contiguous(dats)

def easyplot(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    fig.show()

def easyplot_day(*data):
    fig, ax = plt.subplots()
    for datum in data:
        ax.plot(range(2880), datum)
    fig.show()

def _plot_florida():
    florida = {
        'bkvl', 'brtw', 'chin', 'crst', 'fl75', 'flbn', 'flf1', 'flkw', 'flwe',
        'fmyr', 'gnvl', 'laud', 'mtnt', 'napl', 'okcb', 'ormd', 'pbch', 'pcla',
        'pltk', 'prry', 'talh', 'wach', 'xcty', 'zefr', 'zjx1', 'zma1',
    }
    colors = {'okcb': 'orange', 'zjx1': 'black', 'dlnd': 'blue', 'ormd': 'green', 'flwe': 'pink', 'napl': 'yellow', 'pbch': 'red'}

    fig, ax = plt.subplots()
    added = set()
    for i in range(len(ticks)):
        for entry in ticks[i][2]:
            if entry[1] in added:
                continue
            name = entry[1][:4]
            if name in florida:
                ax.plot(range(i, i+len(filtered[entry[1]])), filtered[entry[1]], label=entry[1], color=colors[name])
            added.add(entry[1])
    #h, l = ax.get_legend_handles_labels()
    ax.legend(h, l)
    fig.show()

def plot_florida():
    stations = {
        'bkvl', 'brtw', 'chin', 'crst', 'fl75', 'flbn', 'flf1', 'flkw', 'flwe',
        'fmyr', 'gnvl', 'laud', 'mtnt', 'napl', 'okcb', 'ormd', 'pbch', 'pcla',
        'pltk', 'prry', 'talh', 'wach', 'xcty', 'zefr', 'zjx1', 'zma1',
    }
    stations &= set(station_data.keys())
    receivers = ['G%02d' % i for i in range(1, 32)]
    combos = itertools.product(stations, receivers)

    easyplot_day(*[get_bpfilt_for(st, sv)[1] for st, sv in combos])

def plot_florida_depletion():
    stations = {
        'bkvl', 'brtw', 'chin', 'crst', 'fl75', 'flbn', 'flf1', 'flkw', 'flwe',
        'fmyr', 'gnvl', 'laud', 'mtnt', 'napl', 'okcb', 'ormd', 'pbch', 'pcla',
        'pltk', 'prry', 'talh', 'wach', 'xcty', 'zefr', 'zjx1', 'zma1',
    }
    stations &= set(station_data.keys())
    receivers = ['G%02d' % i for i in range(1, 32)]
    combos = itertools.product(stations, receivers)
    easyplot_day(*[get_depletion_for(st, sv)[1] for st, sv in combos])

def plot_map_depletion():
    globe = Basemap(projection='mill',lon_0=180)
    # plot coastlines, draw label meridians and parallels.
    globe.drawcoastlines()
    #globe.drawparallels(numpy.arange(-90,90,30),labels=[1,0,0,0])
    #globe.drawmeridians(numpy.arange(globe.lonmin,globe.lonmax+30,60),labels=[0,0,0,1])

    scatter = globe.scatter([], [])

    stations = {
        'bkvl', 'brtw', 'chin', 'crst', 'fl75', 'flbn', 'flf1', 'flkw', 'flwe',
        'fmyr', 'gnvl', 'laud', 'mtnt', 'napl', 'okcb', 'ormd', 'pbch', 'pcla',
        'pltk', 'prry', 'talh', 'wach', 'xcty', 'zefr', 'zjx1', 'zma1',
    }
    stations &= set(station_data.keys())
    receivers = ['G%02d' % i for i in range(1, 32)]
    lats = [[] for _ in range(2880)]
    lons = [[] for _ in range(2880)]
    values = [[] for _ in range(2880)]

    for st, sv in itertools.product(stations, receivers):
        locs, depls = get_depletion_for(st, sv)
        for i in range(2880):
            loc = locs[i]
            depl = depls[i]
            if loc is not None and not math.isnan(depl):
                lat, lon, _ = coordinates.ecef2geodetic(loc)
                lon = lon if lon > 0 else lon + 360
                lats[i].append(lat)
                lons[i].append(lon)
                values[i].append(depl)

    def animate(i):
        plt.title( str("%0.2f" % (24 * i/2880)) )
        lon = lons[i]
        lat = lats[i]
        val = values[i]

        scatter.set_offsets( numpy.array(globe(lon, lat)).T )

        scatter.set_color( cm.plasma(numpy.array(val)/8) )

    def init():
        scatter.set_offsets([])

    ani = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=range(1600, 2400), repeat=True, interval=60)

    #globe.draw()
    #globe.show(block=False)
    plt.show()
    print("done")
    return ani


def plot_map():
    globe = Basemap(projection='mill',lon_0=180)
    # plot coastlines, draw label meridians and parallels.
    globe.drawcoastlines()
    #globe.drawparallels(numpy.arange(-90,90,30),labels=[1,0,0,0])
    #globe.drawmeridians(numpy.arange(globe.lonmin,globe.lonmax+30,60),labels=[0,0,0,1])

    scatter = globe.scatter([], [])

    def animate(i):
        plt.title( str(meas_data[i][0].recv_time.as_datetime()) )
        scatter.set_offsets(numpy.c_[ticks[i][0], ticks[i][1]])
        tec_raw = numpy.array([x[0] for x in ticks[i][2]])

        corrections = numpy.array([smoothed[x[1]][x[2]] for x in ticks[i][2]])

        deviation_from_mean = tec_raw - corrections

        post_filt = numpy.array([filtered[x[1]][x[2]] for x in ticks[i][2]])
        scaled = (post_filt) / (max_tec - min_tec) + 0.5

        for j, val in enumerate(ticks[i][2]):
            stuff[val[1]].append((scaled[j], i))

        scatter.set_color( cm.plasma(scaled) )

    def init():
        scatter.set_offsets([])

    ani = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=range(1740, 2200), repeat=True, interval=160)

    #globe.draw()
    #globe.show(block=False)
    plt.show()
    print("done")
    return ani
