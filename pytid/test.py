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
from scipy import signal
from scipy.signal import butter, lfilter


import tec
import get_data


dog = AstroDog(cache_dir=os.environ['HOME'] + "/.gnss_cache/")

print("initializing coords...")
globe = Basemap(projection='mill',lon_0=180)

print("loading data...")
stations = [
    'flwe', 'ormd', 'dlnd', 'okcb', 'pcla', 'mmd1', 'bmpd', 'okte', 'blom',
    'utmn', 'nvlm', 'p345', 'slac', 'ndst', 'pamm', 'njmt', 'kybo', 'mtlw',
    'scsr', 'cofc', 'nmsu', 'azmp', 'wask', 'dunn', 'zjx1', 'talh', 'gaay',
    'ztl4', 'aldo', 'fmyr', 'pcla', 'crst', 'altu', 'mmd1', 'prjc', 'msin',
    'cola', 'alla', 'mspe', 'tn22', 'tn18', 'wvat', 'ines', 'freo', 'hnpt',
    'flbn', 'scfj', 'ncbx', 'ncdu'
]
station_poss = []
meas_datas = []
for station in stations:
    try:
        #station_pos, meas_data = get_data.data_for_station(dog, station, date=datetime(2019, 12, 17))
        station_pos, meas_data = get_data.data_for_station(dog, station, date=datetime(2020, 2, 17))
        #station_pos, meas_data = get_data.data_for_station(dog, station, date=datetime(2018, 2, 6))
        station_poss.append(station_pos)
        meas_datas.append(meas_data)
    except (ValueError, rinex_file.DownloadError):
        print("*** error with station " + station)

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
    #numpy.array(vals)/400 + 0.5) )

epsilon = 0.4
min_tec = numpy.quantile([x for x in all_values if not math.isnan(x)], 0.5 - epsilon)
max_tec = numpy.quantile([x for x in all_values if not math.isnan(x)], 0.5 + epsilon)

smoothed = dict()
pattern = numpy.ones(5)/5
for name, connection in connections.items():
    smoothed[name] = numpy.convolve(connection, pattern, 'same')
    # ignore the ends
    smoothed[name][:len(pattern)] = math.nan
    smoothed[name][-len(pattern):] = math.nan

# https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


filtered = dict()
b, a = signal.butter(5, .1, btype='high', analog='false')
for name, connection in connections.items():
    filtered[name] = butter_bandpass_filter(connection, 1/(10*60), 1/(4*60), 0.5)


print("max/min = %f/%f" % (max_tec, min_tec))

print("plotting...")

stuff = defaultdict(list)

def plot():
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
