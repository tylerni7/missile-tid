"""
Helpful plotting functions for TID results
"""
from datetime import timedelta
from typing import Iterable, Optional, Tuple

import cartopy
import cartopy.feature as cpf
from matplotlib import animation, cm, pyplot as plt
import numpy

from tid.scenario import Scenario


# the size of TEC waves to look for after filtering, in TECu
TID_SCALE = 0.1


def plot_map(
    scenario: Scenario,
    extent: Optional[Tuple[float, float, float, float]] = None,
    frames: Optional[Iterable[int]] = None,
) -> animation.Animation:
    """
    Plot an animated map of the scenario's filtered VTEC values

    Args:
        scenario: the scenario containing the data we want
        extent: a four-tuple of min longitude, max longitude, min latitude, max latitude
            which defines the boundaries of the graphing region
            if None, defaults to the scenario's default extent
        frames: optional iterable of tick numbers to show

    Returns:
        animation object (in case you want to save a gif)
    """
    ax = plt.axes(projection=cartopy.crs.PlateCarree())
    ax.add_feature(cpf.COASTLINE)
    scatter = ax.scatter([], [])
    if extent is None:
        extent = scenario.get_extent()
    ax.set_extent(extent)

    vtec_map, coord_map = scenario.get_filtered_vtec_data()

    def animate(i):
        plt.title(str(timedelta(seconds=i * 30) + scenario.start_date))

        lons = []
        lats = []
        vals = []
        for station in vtec_map.keys():
            for prn in vtec_map[station].keys():
                if coord_map[station][prn][i] is None:
                    continue

                lat, lon = coord_map[station][prn][i]
                lon = lon % 360  # make sure it's positive, cartopy needs that
                lons.append(lon)
                lats.append(lat)
                vals.append(vtec_map[station][prn][i])

        scatter.set_offsets(numpy.array((lons, lats)).T)
        nvals = numpy.array(vals)
        if len(nvals) > 0:
            # re-center about 0 and clip
            nvals = numpy.clip(nvals + TID_SCALE, 0, TID_SCALE * 2)
            # normalize data from 0 to 1
            nvals /= TID_SCALE * 2
            scatter.set_color(cm.plasma(nvals))

    def init():
        scatter.set_offsets([])

    ani = animation.FuncAnimation(
        plt.gcf(), animate, init_func=init, frames=frames, repeat=True, interval=60
    )
    plt.show()
    return ani
