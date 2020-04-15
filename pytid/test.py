from collections import defaultdict
from datetime import datetime
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

print("initializing coords...")
globe = Basemap(projection='mill',lon_0=180)

print("loading data...")
stations = [
    'napl', 'wach', 'bkvl', 'zefr', 'pbch', 'flwe', 'ormd', 'flbn', 'pcla'
]
"""
    'flwe', 'ormd', 'dlnd', 'okcb', 'pcla', 'mmd1', 'bmpd', 'okte', 'blom',
    'utmn', 'nvlm', 'p345', 'slac', 'ndst', 'pamm', 'njmt', 'kybo', 'mtlw',
    'scsr', 'cofc', 'nmsu', 'azmp', 'wask', 'dunn', 'zjx1', 'talh', 'gaay',
    'ztl4', 'aldo', 'fmyr', 'pcla', 'crst', 'altu', 'mmd1', 'prjc', 'msin',
    'cola', 'alla', 'mspe', 'tn22', 'tn18', 'wvat', 'ines', 'freo', 'hnpt',
    'flbn', 'ncbx', 'ncdu', 'pbch', 'napl'
]
"""
station_poss = []
meas_datas = []

def add_station(station):
    try:
        station_pos, meas_data = get_data.data_for_station(dog, station, date=datetime(2019, 12, 17))
        #station_pos, meas_data = get_data.data_for_station(dog, station, date=datetime(2020, 2, 17))
        #station_pos, meas_data = get_data.data_for_station(dog, station, date=datetime(2018, 2, 6))
        station_poss.append(station_pos)
        meas_datas.append(meas_data)
    except (ValueError, rinex_file.DownloadError):
        print("*** error with station " + station)

for station in stations:
    add_station(station)

print("doing calcs...")

# for each satellite <-> receiver pair, we should have a correction
# just use the average TEC value, I guess? use this to store it
connections = defaultdict(list)
all_values = []

ticks = []
for i in range(2880):
    if i % 100 == 0:
        print(i)
    lats = []
    lons = []
    vals = []

    for j in range(len(meas_datas)):
        if len(meas_datas[j]) <= i:
            continue

        meas = meas_datas[j][i]
        for sv_dat in meas:
            res = tec.calc_vtec(dog, station_poss[j], sv_dat)
            if res is None:
                continue
            vtec, loc = res

            all_values.append(vtec)

            lat, lon, _ = coordinates.ecef2geodetic(loc)
            lats.append(lat)
            lons.append(lon if lon > 0 else lon + 360)
            connection_name = stations[j] + "-" + sv_dat.prn

            vals.append( (vtec, connection_name, len(connections[connection_name])) )
            connections[connection_name].append(vtec)

    xs, ys = globe(lons, lats)
    ticks.append( (xs, ys, vals) )

epsilon = 0.4
min_tec = numpy.quantile([x for x in all_values if not math.isnan(x)], 0.5 - epsilon)
max_tec = numpy.quantile([x for x in all_values if not math.isnan(x)], 0.5 + epsilon)

smoothed = dict()
pattern = numpy.ones(7)/7
for name, connection in connections.items():
    smoothed[name] = numpy.convolve(connection, pattern, 'same')
    # ignore the ends
    smoothed[name][:len(pattern)] = 0
    smoothed[name][-len(pattern):] = 0

# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    a, b = butter_bandpass(lowcut, highcut, fs, order=order)
#    y = lfilter(b, a, data)
    if len(data) < 27:
        return [math.nan] * len(data)
    return filtfilt(a, b, data)


filtered = dict()
for name, connection in connections.items():
    filtered[name] = butter_bandpass_filter(connection, 1/(12*60), 1/(2*60), 1/30)

def get_connection_info(station, sv):
    res = []
    idx = stations.index(station)
    for i, dat in enumerate(meas_datas[idx]):
        for sv_dat in dat:
            if sv_dat.prn == sv:
                res.append(sv_dat)
                break
        else:
            res.append(None)
    return res

stuff = defaultdict(list)

def easyplot(data):
    fig, ax = plt.subplots()
    ax.plot(data)
    fig.show()

def easyplot_day(data):
    fig, ax = plt.subplots()
    ax.plot(range(2880), data)
    fig.show()

def plot_florida():
    florida = {'brtw', 'dlnd', 'flcb', 'flwe', 'okcb', 'ormd', 'pltk', 'zjx1', 'napl', 'pbch'}
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

"""
prjc-G27
prjc-R19
prjc-R20
"""
