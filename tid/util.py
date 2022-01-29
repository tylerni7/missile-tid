"""
Generic utility functions that help make life easier when dealing with data
Should be mostly short wrapper functions
"""
from datetime import datetime, timedelta
from typing import cast, Optional, Sequence

import numpy
from scipy.signal import butter, filtfilt

from laika.gps_time import GPSTime
from laika.lib import coordinates

from tid import types

DATA_RATE = 30  # how many seconds / measurement
DAYS = timedelta(days=1)
HOURS = timedelta(hours=1)


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


def station_location_from_rinex(rinex_path: str) -> Optional[types.ECEF_XYZ]:
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
                xyz = numpy.array([float(x) for x in linedat.split()[:3]])
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
                return cast(types.ECEF_XYZ, xyz)
    return None


def get_dates_in_range(start_date: datetime, duration: timedelta) -> Sequence[datetime]:
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


BUTTER_MIN_LENGTH = 28


def butter_bandpass_filter(
    data: numpy.ndarray,
    lowcut: float,
    highcut: float,
    samplerate: float,
    order: int = 2,
):
    """
    Generic Butterworth bandpass filter function
    https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter

    Args:
        data: 1D numpy array of data at 1 sample per DATA_RATE time
        lowcut: frequency (in Hz) below which to attenuate
        highcut: frequency (in Hz) below which to attenuate
        samplerate: sampling rate frequency (in Hz) of the incoming data
        order: the order of the polynomial or whatever to use for filtering the data

    Returns:
        1D numpy array of the filtered data, or None if there wasn't enough data to properly filter
    """
    nyq = 0.5 * samplerate
    lowf = lowcut / nyq
    highf = highcut / nyq
    # generic names for coefficients in filters
    # pylint: disable=invalid-name
    a, b = butter(order, [lowf, highf], btype="band")
    if len(data) < BUTTER_MIN_LENGTH:
        return None
    return filtfilt(a, b, data)


def bpfilter(
    data: numpy.ndarray, short_min: float = 2, long_min: float = 12
) -> Optional[numpy.ndarray]:
    """
    Perform a 2nd-order Butterworth Bandpass filter on the given data

    Args:
        data: 1D numpy array of data at 1 sample per DATA_RATE time
        short_min: attenuate signals with periods below this many minutes
        long_min: attenuate signals with periods above this many minutes

    Returns:
        1D numpy array of the filtered data, or None if there wasn't enough data to properly filter
    """
    return butter_bandpass_filter(
        data, 1 / (long_min * 60), 1 / (short_min * 60), 1 / DATA_RATE
    )


def segmenter(data_stream: numpy.ndarray) -> Sequence[int]:
    """
    Split up a signal that should be "steady"
    Return a list of all indices which should represent
    boundaries and be tossed out

    Args:
        data_stream: numpy array of 1d data that should be ~constant

    Returns:
        list of indices to remove
    """
    diff = numpy.median(
        numpy.convolve(
            numpy.abs(numpy.diff(data_stream)), numpy.array([1, 1, 1, 1, 1]) / 5
        )
    )
    return cast(
        Sequence[int],
        numpy.where(
            numpy.abs(numpy.diff(data_stream, prepend=data_stream[0])) > diff * 5
        )[0],
    )
