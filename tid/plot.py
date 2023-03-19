"""
Helpful plotting functions for TID results
"""
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cartopy
import cartopy.feature as cpf
from matplotlib import animation, cm, pyplot as plt
import numpy

from tid.scenario import Scenario


# the size of TEC waves to look for after filtering, in TECu
TID_SCALE = 0.1


def plot_filtered_vtec(scenario: Scenario, station: str, prn: str):
    """
    Plot the TEC after being bandpass filtered

    Args:
        scenario: the scenario with the data
        station: the station we want data for
        prn: the satellite we want data for
    """
    fig, axis = plt.subplots()
    axis.plot(scenario.conn_map[station][prn].get_filtered_vtecs())
    plt.title(f"Station: {station} Satellite: {prn}")
    plt.ylabel("vTEC")
    plt.xlabel("Time (ticks)")
    fig.show()


def plot_raw_vtec(scenario: Scenario, station: str, prn: str):
    """
    Plot the TEC with no filtering

    Args:
        scenario: the scenario with the data
        station: the station we want data for
        prn: the satellite we want data for
    """
    fig, axis = plt.subplots()
    axis.plot(scenario.conn_map[station][prn].get_vtecs())
    plt.title(f"Station: {station} Satellite: {prn}")
    plt.ylabel("vTEC")
    plt.xlabel("Time (ticks)")
    fig.show()


def plot_map(
    scenario: Scenario,
    extent: Optional[Tuple[float, float, float, float]] = None,
    frames: Optional[Iterable[int]] = None,
    raw: bool = False,
    display: bool = True,
) -> animation.Animation:
    """
    Plot an animated map of the scenario's filtered VTEC values

    Args:
        scenario: the scenario containing the data we want
        extent: a four-tuple of min longitude, max longitude, min latitude, max latitude
            which defines the boundaries of the graphing region
            if None, defaults to the scenario's default extent
        frames: optional iterable of tick numbers to show
        raw: whether to plot raw vtec data or filtered
        display: whether to show the animation or not

    Returns:
        animation object (in case you want to save a gif)
    """
    axis = plt.axes(projection=cartopy.crs.PlateCarree())
    axis.clear()
    axis.add_feature(cpf.COASTLINE.with_scale("10m"))
    axis.add_feature(cpf.BORDERS.with_scale("10m"), edgecolor="gray", linewidth=0.3)

    scatter = axis.scatter([], [])
    title = plt.title("Date")
    if extent is None:
        extent = scenario.get_extent()
    axis.set_extent(extent)

    vtec_map, coord_map = scenario.get_vtec_data(raw=raw)

    def animate(i):
        title.set_text(str(timedelta(seconds=i * 30) + scenario.start_date) + " UTC")

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
        # scale = (0, 25) if raw else (-TID_SCALE, TID_SCALE)
        scale = (20, 30) if raw else (-TID_SCALE, TID_SCALE)
        nvals = numpy.array(vals)
        if len(nvals) > 0:
            # re-center about 0 and clip
            nvals = numpy.clip(nvals - scale[0], 0, scale[1] - scale[0])
            # normalize data from 0 to 1
            nvals /= scale[1] - scale[0]
            scatter.set_color(cm.plasma(nvals))

    def init():
        return scatter

    ani = animation.FuncAnimation(
        plt.gcf(),
        animate,
        init_func=init,
        frames=frames,
        repeat=True,
        interval=60,
    )
    axis.figure.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    axis.figure.subplots_adjust(
        left=0, bottom=0, right=1, top=1, wspace=None, hspace=None
    )
    if display:
        plt.show()
    return ani


def save_plot(anim: animation.Animation, name: str, path: Path) -> None:
    """
    Plot an animated map of the scenario's filtered VTEC values

    Args:
        anim: The matplotlib animation to save
        name: The name of the animation
        path: The directory in which to save the animation
    """
    anim.save((path / f"{name}.gif").as_posix(), writer="imagemagick", fps=60)
