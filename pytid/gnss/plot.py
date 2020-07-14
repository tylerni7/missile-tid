from datetime import datetime, timedelta
from laika.lib import coordinates
import logging
import math
from matplotlib import animation, cm, pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy
import os

from pytid.gnss import get_data
from pytid.utils.configuration import Configuration


config = Configuration()
constellation_size = config.gnss.get("constellation_size")
plot_dir = config.plotting.get("output_dir")
logger = logging.getLogger(__name__)


class StationPlotter:
    """
    Helper class for the plotting of the TEC at each station
    """

    def __init__(self, vtecs, date: datetime, to_disk: bool = True, filename_suffix = None):
        """
        :param vtecs: The corrected vertical total electron content data
        :param to_disk: Save the plot to disk
        """
        self.fig = None
        self.ax = None
        self.vtecs = vtecs
        self.date = date
        self.to_disk = to_disk
        if filename_suffix is None:
            self.filename_suffix = ''
        else:
            self.filename_suffix = '_' + filename_suffix

    def finish_plots(self, station, suffix = None):
        if suffix is None:
            if self.filename_suffix is not None:
                mysuffix = self.filename_suffix
            else:
                mysuffix = ''
        else:
            mysuffix = '_' + str(suffix)
        # Let's not cause fights between Americans and other nations about ambiguous time formats :-)
        readable_date = self.date.strftime("%b %d %Y")
        numeric_date = self.date.strftime("%Y_%m_%d")
        plt.title(f"Station: {station} Date: {readable_date}")
        plt.ylabel("vTEC")
        plt.xlabel("Time")

        if self.to_disk:
            os.makedirs(plot_dir, exist_ok=True)
            path = os.path.join(plot_dir, f"{numeric_date}_{station}{mysuffix}.png")
            logger.info(f"Saving plot to {path}")
            plt.savefig(path)

        self.fig.show()

    def plot_station(self, station):
        self.fig, self.ax = plt.subplots()
        for sat_name in get_data.satellites:
            if sat_name not in self.vtecs[station]:
                continue
            self.ax.plot([(x if x else math.nan) for x in self.vtecs[station][sat_name][1]])
        self.finish_plots(station)

    def plot_station_depletion(self, station):
        self.fig, self.ax = plt.subplots()
        for sat_name in get_data.satellites:
            if sat_name not in self.vtecs[station]:
                continue
            self.ax.plot(get_data.depletion_contiguous(
                [(x if x else math.nan) for x in self.vtecs[station][sat_name][1]]
            ))
        self.finish_plots(station)

    def plot_station_filter(self, station, short_min=2, long_min=12):
        self.fig, self.ax = plt.subplots()
        for sat_name in get_data.satellites:
            if sat_name not in self.vtecs[station]:
                continue
            self.ax.set_ylim(-0.5, 0.5)
            self.ax.plot(
                get_data.filter_contiguous(
                    [(x if x else math.nan) for x in self.vtecs[station][sat_name][1]],
                    short_min=short_min, long_min=long_min
                ),
                scaley=False
            )
        self.finish_plots(station)



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
            for prn in get_data.satellites:
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
        max_tec = 20
        min_tec = 8
        scatter.set_color( cm.plasma(numpy.maximum(numpy.array(vals) - min_tec, 0) / (max_tec - min_tec)) )

    def init():
        scatter.set_offsets([])

    ani = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=frames, repeat=True, interval=60)

    #globe.draw()
    #globe.show(block=False)
    plt.show()
    return ani

def plot_filter_map(vtec, stations, start_date, short_min=2, long_min=12, frames=None):
    globe = Basemap(projection='mill',lon_0=180)
    # plot coastlines, draw label meridians and parallels.
    globe.drawcoastlines()
    #globe.drawparallels(numpy.arange(-90,90,30),labels=[1,0,0,0])
    #globe.drawmeridians(numpy.arange(globe.lonmin,globe.lonmax+30,60),labels=[0,0,0,1])

    scatter = globe.scatter([], [])

    print("filtering")
    filtered = dict()
    for station in stations:
        filtered[station] = dict()
        for prn in get_data.satellites:
            if prn not in vtec[station]:
                continue
            filtered[station][prn] = get_data.filter_contiguous(
                [(x if x else math.nan) for x in vtec[station][prn][1]],
                short_min=short_min,
                long_min=long_min
            )
    print("filtering done")

    #ttext = plt.text(2.5e7, 1.98e7, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5})
    def animate(i):
        plt.title( str(timedelta(seconds=i * 30) + start_date) )

        lons = []
        lats = []
        vals = []
        for station in stations:
            for prn in get_data.satellites:
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

                vals.append(filtered[station][prn][i])

        scatter.set_offsets(numpy.array(globe(lons, lats)).T)
        max_tec = 0.1
        min_tec = -0.1
        scatter.set_color( cm.plasma(numpy.maximum(numpy.array(vals) - min_tec, 0) / (max_tec - min_tec)) )

    def init():
        scatter.set_offsets([])

    ani = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=frames, repeat=True, interval=60)

    #globe.draw()
    #globe.show(block=False)
    plt.show()
    return ani