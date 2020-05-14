from datetime import timedelta
from matplotlib import animation
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy

from laika.lib import coordinates

from . import get_data

def plot_station(vtecs, station):
    fig, ax = plt.subplots()
    for i in range(1, 33): 
        if 'G%02d' % i not in vtecs[station]:
            continue
        ax.plot([(x if x else math.nan) for x in vtecs[station]['G%02d' % i][1]])
    fig.show()

def plot_station_depletion(vtecs, station):
    fig, ax = plt.subplots()
    for i in range(1, 33): 
        if 'G%02d' % i not in vtecs[station]:
            continue
        ax.plot(get_data.depletion_contiguous(
            [(x if x else math.nan) for x in vtecs[station]['G%02d' % i][1]]
        ))
    fig.show()

def plot_station_filter(vtecs, station, short_min=2, long_min=12):
    fig, ax = plt.subplots()
    for i in range(1, 33): 
        if 'G%02d' % i not in vtecs[station]:
            continue
        ax.plot(get_data.filter_contiguous(
            [(x if x else math.nan) for x in vtecs[station]['G%02d' % i][1]],
            short_min=short_min, long_min=long_min
        ))
    fig.show()


def plot_map(vtec, stations, start_date, frames=None):
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

    ani = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=frames, repeat=True, interval=60)

    #globe.draw()
    #globe.show(block=False)
    plt.show()
    return ani
