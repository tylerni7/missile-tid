"""
Generic utility functions that help make life easier when dealing with data
Should be mostly short wrapper functions
"""
from datetime import datetime, timedelta
from typing import Optional, Sequence, Iterable
import numpy

from laika.gps_time import GPSTime
from laika.lib import coordinates

DAYS = timedelta(days=1)


def gpstime_fromstr(timestr: str) -> GPSTime:
    """
    Give a laika GPSTime object for the given time string

    Args:
        timestr: string like "2020-01-30" indicating the date

    Returns:
        GPSTime object for the same date
    """

    return GPSTime.from_datetime(datetime.strptime(timestr, "%Y-%m-%d"))


def datetime_fromstr(timestr: str) -> GPSTime:
    """
    Give a datetime object for the given time string

    Args:
        timestr: string like "2020-01-30" indicating the date

    Returns:
        datetime object for the same date
    """

    return datetime.strptime(timestr, "%Y-%m-%d")


def channel2(observations: numpy.array) -> str:
    """
    Frequently we want to know if the channel 2 code phase data
    is from C2C or C2P. This function wraps that (simple) logic
    to keep things cleaner

    Args:
        observations: the numpy array of dense observations

    Returns:
        a string of "C2C" or "C2P"

    Raises:
        LookupError if neither of those signals is available
    """
    # default channel 2 code phase signal
    chan2 = "C2C"
    if numpy.isnan(observations[0]["C2C"]):
        # less reliable channel 2 code phase signal
        chan2 = "C2P"
        if numpy.isnan(observations[0]["C2P"]):
            # if we don't have that, we're done
            raise LookupError
    return chan2


def station_location_from_rinex(rinex_path: str) -> Optional[Sequence]:
    """
    Opens a RINEX file and looks in the headers for the station's position

    Args:
        rinex_path: the path to the rinex file

    Returns:
        XYZ ECEF coords in meters for the approximate receiver location
        approximate meaning may be off by a meter or so
        or None if ECEF coords could not be found
    """

    xyz = None
    lat = None
    lon = None
    height = None
    with open(rinex_path, "rb") as filedat:
        for _ in range(50):
            linedat = filedat.readline()
            if b"POSITION XYZ" in linedat:
                xyz = [float(x) for x in linedat.split()[:3]]
            elif b"Monument location:" in linedat:
                lat, lon, height = [float(x) for x in linedat.split()[2:5]]
            elif b"(latitude)" in linedat:
                lat = float(linedat.split()[0])
            elif b"(longitude)" in linedat:
                lon = float(linedat.split()[0])
            elif b"(elevation)" in linedat:
                height = float(linedat.split()[0])

            if lat is not None and lon is not None and height is not None:
                xyz = coordinates.geodetic2ecef((lat, lon, height))

            if xyz is not None:
                return xyz
    return None


def get_dates_in_range(start_date: datetime, duration: timedelta) -> Iterable[datetime]:
    """
    Get a list of dates, starting with start_date, each 1 day apart

    Args:
        start_date: the first date to include
        duration: how long to include

    Returns:
        list of dates, each separated by 1 day
    """
    first_day = start_date.replace(hour=0, minute=0)
    dates = [first_day]
    last_date = first_day + timedelta(days=1)
    deadline = start_date + duration
    while last_date < deadline:
        dates.append(last_date)
        last_date += timedelta(days=1)
    return dates
