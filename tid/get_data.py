"""
Functions to help with download and basic processing of GPS data
"""
from datetime import datetime
import io
import json
import logging
import os
import re
from typing import Iterable, Optional, Sequence
import zipfile

import requests
import numpy

from laika import AstroDog, raw_gnss
from laika.dgps import get_station_position
from laika.downloader import download_cors_station, download_and_cache_file
from laika.gps_time import GPSTime
from laika.raw_gnss import GNSSMeasurement
from laika.rinex_file import RINEXFile, DownloadError

from tid import util


LOG = logging.getLogger(__name__)

# ecef locations for stations, so we can know what is nearby
with open(
    os.path.dirname(__file__) + "/lookup_tables/station_locations.json", "rb"
) as f:
    STATION_LOCATIONS = json.load(f)

# which network stations belong to, if we know, to speed up downloading
with open(
    os.path.dirname(__file__) + "/lookup_tables/station_networks.json", "rb"
) as f:
    STATION_NETWORKS = json.load(f)


def get_nearby_stations(
    dog: AstroDog, point: Sequence, dist: int = 400000
) -> Sequence[str]:
    """
    Find all known/downloadable station names within a given distance from
    the target point.

    Args:
        dog: laika AstroDog object
        point: tuple of ECEF xyz location, in meters
        dist: allowable distance from the target point, in meters

    Returns:
        a list of strings representing station names close to the target point
    """
    cache_dir = dog.cache_dir
    cors_pos_path = cache_dir + "cors_coord/cors_station_positions"
    with open(cors_pos_path, "rb") as cors_pos:
        # pylint:disable=unexpected-keyword-arg
        # (confused about numpy, I guess)
        cors_pos_dict = numpy.load(cors_pos, allow_pickle=True).item()
    station_names = []
    station_pos = []

    for name, (_, pos, _) in cors_pos_dict.items():
        station_names.append(name)
        station_pos.append(pos)
    for name, pos in STATION_LOCATIONS.items():
        station_names.append(name)
        station_pos.append(pos)

    station_names = numpy.array(station_names)
    station_pos = numpy.array(station_pos)
    point = numpy.array(point)

    dists = numpy.sqrt(((station_pos - numpy.array(point)) ** 2).sum(1))

    return list(station_names[numpy.where(dists < dist)[0]])


def _download_misc_igs_station(
    dog: AstroDog, time: GPSTime, station_name: str
) -> Optional[str]:
    """
    Downloader for non-CORS stations. Attempts to download rinex observables
    for the given station and time
    Should only be used internally by data_for_station

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object
        station_name: string representation a station name

    Returns:
        string representing a path to the downloaded file
        or None, if the file was not able to be downloaded
    """
    cache_subdir = dog.cache_dir + "misc_igs_obs/"
    t = time.as_datetime()
    # different path formats...

    folder_path = t.strftime("%Y/%j/")
    filename = station_name + t.strftime("%j0.%yo")
    url_bases = (
        "ftp://garner.ucsd.edu/archive/garner/rinex/",
        "ftp://data-out.unavco.org/pub/rinex/obs/",
    )
    try:
        filepath = download_and_cache_file(
            url_bases, folder_path, cache_subdir, filename, compression=".Z"
        )
        return filepath
    except IOError:
        url_bases = (
            "ftp://igs.gnsswhu.cn/pub/gps/data/daily/",
            "ftp://cddis.nasa.gov/gnss/data/daily/",
        )
        folder_path += t.strftime("%yo/")
        try:
            filepath = download_and_cache_file(
                url_bases, folder_path, cache_subdir, filename, compression=".Z"
            )
            return filepath
        except IOError:
            return None


def _download_korean_station(
    dog: AstroDog, time: GPSTime, station_name: str
) -> Optional[str]:
    """
    Downloader for Korean stations. Attempts to download rinex observables
    for the given station and time.
    Should only be used internally by data_for_station

    TODO: we can download from multiple stations at once and save some time here....

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object
        station_name: string representation a station name

    Returns:
        string representing a path to the downloaded file
        or None, if the file was not able to be downloaded
    """
    json_url = "http://gnssdata.or.kr/download/createToZip.json"
    zip_url = "http://gnssdata.or.kr/download/getZip.do?key=%d"

    cache_subdir = dog.cache_dir + "korean_obs/"
    t = time.as_datetime()
    # different path formats...
    folder_path = cache_subdir + t.strftime("%Y/%j/")
    filename = folder_path + station_name + t.strftime("%j0.%yo")

    if os.path.isfile(filename):
        return filename
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    start_day = t.strftime("%Y%m%d")
    postdata = {
        "corsId": station_name.upper(),
        "obsStDay": start_day,
        "obsEdDay": start_day,
        "dataTyp": 30,
    }
    res = requests.post(json_url, data=postdata).text
    if not res:
        raise DownloadError
    res_dat = json.loads(res)
    if not res_dat.get("result", None):
        raise DownloadError

    key = res_dat["key"]
    zipstream = requests.get(zip_url % key, stream=True)
    with zipfile.ZipFile(io.BytesIO(zipstream.content)) as zipdat:
        for zipf in zipdat.filelist:
            with zipfile.ZipFile(io.BytesIO(zipdat.read(zipf))) as station:
                for rinex in station.filelist:
                    if rinex.filename.endswith("o"):
                        with open(filename, "wb") as rinex_out:
                            rinex_out.write(station.read(rinex))
    return filename


def cors_get_station_lists_for_day(date: datetime) -> Iterable[str]:
    """
    Given a date, returns the stations that the US CORS network
    reports as available
    """
    url = "https://geodesy.noaa.gov/corsdata/rinex/"
    resp = requests.get(url + date.strftime("%Y/%j/"))

    pat = '<a href="..../">([a-z0-9]{4})/</a>'
    return re.findall(pat, resp.text)


def rinex_file_for_station(
    dog: AstroDog, time: GPSTime, station_name: str
) -> Optional[str]:
    """
    Given a particular time and station, get the rinex obs file that
    corresponds to it

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object for the time in question
        station_name: string of the station in question
            station names are CORS names or similar (eg: 'slac')

    Returns:
        the string containing the file path, or None
    """
    rinex_obs_file = None

    # handlers for specific networks
    handlers = {"Korea": _download_korean_station}

    network = STATION_NETWORKS.get(station_name, None)

    # no special network, so try using whatever
    if network is None:
        # step 1: get the station rinex data
        try:
            rinex_obs_file = download_cors_station(
                time, station_name, cache_dir=dog.cache_dir
            )
        except (KeyError, DownloadError):
            # station position not in CORS map, try another thing
            if station_name in STATION_LOCATIONS:
                rinex_obs_file = _download_misc_igs_station(dog, time, station_name)
            else:
                return None

    else:
        rinex_obs_file = handlers[network](dog, time, station_name)

    return rinex_obs_file


def location_for_station(dog: AstroDog, time: GPSTime, station_name: str) -> Sequence:
    """
    Get location for a particular station at a particular time.
    Time is needed so we can look at RINEX files and sanity check
    the location data.

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object for the time in question
        station_name: string of the station in question
            station names are CORS names or similar (eg: 'slac')

    Returns:
        aproximate x,y,z location in ECEF meters

    Raises:
        DownloadError if the RINEX could not be fetched
    """
    rinex_obs_file = rinex_file_for_station(dog, time, station_name)
    if rinex_obs_file is None:
        raise DownloadError

    # start with most accurate positions (from known databases)
    approx_position = util.station_location_from_rinex(rinex_obs_file)
    try:
        station_pos = get_station_position(station_name, cache_dir=dog.cache_dir)
    except KeyError:
        station_pos = numpy.array(
            STATION_LOCATIONS.get(station_name) or approx_position
        )

    # while databases are more accurate, there are some cases of name collsions
    # (eg Korea and US CORS may pick same 4 letter name). To resolve this, favor
    # positions reported from RINEX files if there is a big (>100m) divergence
    if station_pos is not None and approx_position is not None:
        if numpy.linalg.norm(station_pos - approx_position) > 100:
            LOG.warning(
                "for station %s, we have large differences in position reports",
                station_name,
            )
        station_pos = approx_position

    return station_pos


def data_for_station(
    dog: AstroDog, time: GPSTime, station_name: str
) -> Sequence[Sequence[GNSSMeasurement]]:
    """
    Get data from a particular station and time. Wraps a number of laika function calls.

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object for the time in question
        station_name: string of the station in question
            station names are CORS names or similar (eg: 'slac')

    Returns:
        raw_rinex data

    Raises:
        DownloadError if the data could not be fetched
    """
    rinex_obs_file = rinex_file_for_station(dog, time, station_name)
    if rinex_obs_file is None:
        raise DownloadError

    obs_data = RINEXFile(rinex_obs_file, rate=30)
    return raw_gnss.read_rinex_obs(obs_data)
