"""
Functions to help with download and basic processing of GPS data
"""
from datetime import datetime, timedelta
import io
import json
import logging
import os
import re
from typing import cast, Dict, Iterable, Optional, Sequence, Tuple
import zipfile

import numpy
import requests

import georinex

from laika import AstroDog, raw_gnss
from laika.dgps import get_station_position
from laika.downloader import download_cors_station, download_and_cache_file
from laika.gps_time import GPSTime
from laika.raw_gnss import GNSSMeasurement
from laika.rinex_file import RINEXFile, DownloadError

from tid import tec, types, util


LOG = logging.getLogger(__name__)

DENSE_TYPE = [
    ("tick", "i4"),  # tick number the observation was made
    ("C1C", "f8"),  # GNSS measurements, if available
    ("C2C", "f8"),
    ("L1C", "f8"),
    ("L2C", "f8"),
    ("sat_pos", "3f8"),  # satellite position XYZ ECEF in meters
]

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

    np_station_names = numpy.array(station_names)
    np_station_pos = numpy.array(station_pos)

    dists = numpy.sqrt(((np_station_pos - numpy.array(point)) ** 2).sum(1))

    return list(np_station_names[numpy.where(dists < dist)[0]])


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
        "dataTyp": util.DATA_RATE,
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


def location_for_station(
    dog: AstroDog, time: GPSTime, station_name: str
) -> types.ECEF_XYZ:
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


def from_xarray(xarray, start_date: GPSTime) -> types.DenseMeasurements:
    """
    Convert the georinex xarray for a satellite to DenseMeasurements

    Args:
        xarray: the georinex xarray thing
        start_date: time at which tick 0 occurred

    Returns:
        DenseMeasurements for the satellite
    """
    # truncate to observations with data
    xarray = xarray.dropna("time", how="all", subset=["C1"])
    outp = numpy.zeros(xarray.dims["time"], dtype=DENSE_TYPE)

    obs_map = {"C1C": "C1", "C2C": "C2", "C2P": "P2", "L1C": "L1", "L2C": "L2"}
    for obs in ["C1C", "C2C", "L1C", "L2C"]:
        # if the channel doesn't exist, set to NaN
        if obs_map[obs] not in xarray:
            outp[obs][:] = numpy.nan
        else:
            outp[obs][:] = xarray[obs_map[obs]]

    # if the C2C channel is empty/crap, replace it with C2P
    if numpy.all(numpy.isnan(outp["C2C"])):
        outp["C2C"][:] = xarray[obs_map["C2P"]]

    timedeltas = xarray["time"].astype(numpy.datetime64).to_numpy() - numpy.datetime64(
        start_date.as_datetime()
    )
    outp["tick"] = (timedeltas / numpy.timedelta64(util.DATA_RATE, "s")).astype(int)
    return outp


def data_for_station(
    dog: AstroDog,
    time: GPSTime,
    station_name: str,
    start_date: GPSTime,
) -> types.DenseMeasurements:
    """
    Get data from a particular station and time. Wrapper for data_for_station
    inside of get_data

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object for the time in question

    Returns:
        dense raw gps data

    Raises:
        DownloadError if the data could not be fetched

    TODO: caching of the results on disk? or should that happen later?
    """
    rinex_obs_file = rinex_file_for_station(dog, time, station_name)
    if rinex_obs_file is None:
        raise DownloadError

    rinex = georinex.load(rinex_obs_file, interval=30)

    sv_dict_out: Dict[str, types.DenseDataType] = {}
    for sv in rinex.sv.to_numpy():
        sv_dict_out[sv] = from_xarray(rinex.sel(sv=sv), start_date)
    return sv_dict_out


def populate_sat_info(
    dog: AstroDog,
    start_time: GPSTime,
    duration: timedelta,
    station_dict: types.StationPrnMap[types.DenseMeasurements],
) -> None:
    """
    Populate the satellite locations for our measurements

    Args:
        dog: laika AstroDog to use
        start_time: when the 0th tick occurs
        duration: how long until the last tick
        station_dict: mapping to the DenseMeasurements that need correcting
    """

    satellites = {sat: idx for idx, sat in enumerate(dog.get_all_sat_info(start_time))}
    tick_count = int(duration.total_seconds() / util.DATA_RATE)
    # get an accurate view of the satellites at 30 second intervals
    sat_info = numpy.zeros(
        (len(satellites), tick_count), dtype=[("pos", "3f8"), ("vel", "3f8")]
    )

    for tick in range(tick_count):
        tick_info = dog.get_all_sat_info(start_time + util.DATA_RATE * tick)
        for sv, info in tick_info.items():
            sat_info[satellites[sv]][tick] = (info[0], info[1])

    bad_datas = set()
    for station in station_dict:
        for sat in station_dict[station]:
            if sat not in satellites:
                # no info for this satellite, probably not orbiting, remove it
                bad_datas.add((station, sat))
                continue
            ticks = station_dict[station][sat]["tick"]
            time_delays = station_dict[station][sat]["C1C"] / tec.C
            delta_pos = (
                sat_info[satellites[sat]]["vel"][ticks] * time_delays[:, numpy.newaxis]
            )
            corrected_pos = sat_info[satellites[sat]]["pos"][ticks] - delta_pos
            station_dict[station][sat]["sat_pos"][:] = corrected_pos

    for station, sat in bad_datas:
        del station_dict[station][sat]


def merge_data(
    data1: types.DenseMeasurements, data2: types.DenseMeasurements
) -> types.DenseMeasurements:
    """
    Merges two sets of dense measurements together

    Args:
        data1: the first (chronologically) set of data
        data2: the second (chronologically) set of data

    Returns:
        the combined data
    """
    combined = data1.copy()
    for prn in data2:
        # prn only has data in the second dataset
        if prn not in data1:
            combined[prn] = data2[prn]
        # otherwise we need an actual merge
        else:
            combined[prn] = numpy.append(data1[prn], data2[prn])

    return combined


def populate_data(
    stations: Iterable[str],
    start_date: GPSTime,
    duration: timedelta,
    dog: AstroDog,
) -> Tuple[Dict[str, types.ECEF_XYZ], types.StationPrnMap[types.DenseDataType]]:
    """
    Download/populate the station data and station location info

    Args:
        stations: list of station names
        date_list: ordered list of the dates for which to fetch data
        dog: astro dog to use

    Returns:
        dictionary of station names to their locations,
        dictionary of station names to sat names to their dense data

    TODO: is this a good place to be caching results?
    """

    # dict of station names -> XYZ ECEF locations in meters
    station_locs: Dict[str, types.ECEF_XYZ] = {}
    # dict of station names -> dict of prn -> numpy observation data
    station_data = cast(types.StationPrnMap[types.DenseDataType], {})

    for station in stations:
        gps_date = start_date
        while gps_date < start_date + duration.total_seconds():
            tick = int((gps_date - start_date) // util.DATA_RATE)
            try:
                latest_data = data_for_station(
                    dog, gps_date, station, start_date=start_date
                )
            except DownloadError:
                continue
            finally:
                gps_date += (1 * util.DAYS).total_seconds()
            if station not in station_data:
                station_data[station] = latest_data
            else:
                # we've already got some data, so merge it together
                station_data[station] = merge_data(station_data[station], latest_data)

        # didn't download data, ignore it
        if station not in station_data:
            continue

        if station not in station_locs:
            station_locs[station] = location_for_station(dog, gps_date, station)

    populate_sat_info(dog, start_date, duration, station_data)

    return station_locs, station_data
