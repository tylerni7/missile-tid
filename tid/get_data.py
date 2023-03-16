"""
Functions to help with download and basic processing of GPS data
"""
from datetime import datetime, timedelta
import io
import json
import logging
import multiprocessing
import os
import re
from typing import cast, Dict, Iterable, Optional, Sequence, Tuple
import zipfile
from laika.constants import SECS_IN_DAY

import numpy
import requests

import hatanaka
import georinex
import xarray

from laika import AstroDog
from laika.dgps import get_station_position
from laika.downloader import download_cors_station, download_and_cache_file
from laika.gps_time import GPSTime
from laika.rinex_file import DownloadError

from tid import config, tec, types, util


LOG = logging.getLogger(__name__)

DENSE_TYPE = [
    ("tick", "i4"),  # tick number the observation was made
    ("C1C", "f8"),  # GNSS measurements, if available
    ("C2C", "f8"),
    ("L1C", "f8"),
    ("L2C", "f8"),
    ("sat_pos", "3f8"),  # satellite position XYZ ECEF in meters
]

DOWNLOAD_WORKERS = 20  # how many processes to spawn for downloading files

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

conf = config.Configuration()


def char_code_for_partial(time: GPSTime) -> str:
    """
    Preliminary (hourly) data uses a letter to indicate which hour it is for.
    This gets the right code for a given time.

    Args:
        time: the GPSTime start for the time we want

    Returns:
        letter a-x
    """
    return chr(ord("a") + int((time.tow / (60 * 60)) % 24))


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
    station_names = []
    station_pos = []
    for name, pos in STATION_LOCATIONS.items():
        station_names.append(name)
        station_pos.append(pos)

    np_station_names = numpy.array(station_names)
    np_station_pos = numpy.array(station_pos)

    dists = numpy.sqrt(((np_station_pos - numpy.array(point)) ** 2).sum(1))

    return list(np_station_names[numpy.where(dists < dist)[0]])


def _download_misc_igs_station(
    dog: AstroDog, time: GPSTime, station_name: str, partial: bool = False
) -> Optional[str]:
    """
    Downloader for non-CORS stations. Attempts to download rinex observables
    for the given station and time
    Should only be used internally by data_for_station

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object
        station_name: string representation a station name
        partial: whether to get "partial" (hourly) data

    Returns:
        string representing a path to the downloaded file
        or None, if the file was not able to be downloaded
    """
    if partial:
        raise NotImplementedError

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
    dog: AstroDog, time: GPSTime, station_name: str, partial: bool = False
) -> Optional[str]:
    """
    Downloader for Korean stations. Attempts to download rinex observables
    for the given station and time.
    Should only be used internally by data_for_station

    TODO: we can download from multiple stations at once and save some time here....
    TODO: separate network: ftp://gnss-ftp.kasi.re.kr and ftp://nfs.kasi.re.kr (IGS only?) and
        https://gnss.eseoul.go.kr/timeselection

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object
        station_name: string representation a station name
        partial: whether to get "partial" (hourly) data

    Returns:
        string representing a path to the downloaded file
        or None, if the file was not able to be downloaded
    """
    if partial:
        raise NotImplementedError

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


def _download_japanese_station(
    dog: AstroDog, time: GPSTime, station_name: str, partial: bool = False
) -> Optional[str]:
    """
    Downloader for Japanese stations. Attempts to download rinex observables
    for the given station and time.
    Should only be used internally by data_for_station

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object
        station_name: string representation a station name
        partial: whether to get "partial" (hourly) data

    Returns:
        string representing a path to the downloaded file
        or None, if the file was not able to be downloaded
    """
    cache_subdir = dog.cache_dir + "japanese_obs/"
    t = time.as_datetime()
    # different path formats...
    folder_path = t.strftime("%Y/%j/")

    if partial:
        # 'a' = 0, increment by one for each hour
        timecode = char_code_for_partial(time)
    else:
        timecode = "0"
    filename = station_name + t.strftime(f"%j{timecode}.%yo")

    url_bases = ("https://copyfighter.org:6670/japan/data/GR_2.11/",)
    try:
        filepath = download_and_cache_file(
            url_bases, folder_path, cache_subdir, filename, compression=".gz"
        )
        return filepath
    except IOError:
        return None


mongolian_csrf_info = {}


def _get_mongolian_csrf() -> None:
    """
    We need a CSRF token to download things. This will load the page and populate
    the tokens to be used
    """
    req = requests.get("http://monpos.gazar.gov.mn/monstatic")
    mongolian_csrf_info["csrftoken"] = req.cookies["csrftoken"]

    idx = req.text.index('value="', req.text.index("csrfmiddlewaretoken"))
    mongolian_csrf_info["csrfmiddlewaretoken"] = req.text[idx + 7 : idx + 7 + 64]


def _download_mongolian_station(
    dog: AstroDog, time: GPSTime, station_name: str, partial: bool = False
) -> Optional[str]:
    """
    Downloader for Mongolian stations. Attempts to download rinex observables
    for the given station and time.
    Should only be used internally by data_for_station

    Args:
        dog: laika AstroDog object
        time: laika GPSTime object
        station_name: string representation a station name
        partial: whether to get "partial" (hourly) data

    Returns:
        string representing a path to the downloaded file
        or None, if the file was not able to be downloaded
    """
    if partial:
        raise NotImplementedError

    cache_subdir = dog.cache_dir + "mongolian_obs/"
    t = time.as_datetime()
    # different path formats...
    folder_path = cache_subdir + t.strftime("%Y/%j/")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    mongolian_csrf_info
    if not mongolian_csrf_info:
        _get_mongolian_csrf()

    datestr = t.strftime("%Y-%m-%d")
    req = requests.post(
        "http://monpos.gazar.gov.mn/download/" + station_name,
        data={
            "csrfmiddlewaretoken": mongolian_csrf_info["csrfmiddlewaretoken"],
            "datepicker": datestr,
        },
        cookies={"csrftoken": mongolian_csrf_info["csrftoken"]},
    )
    if req.status_code != 200:
        return None

    disk_path = folder_path + station_name + t.strftime(".%yo")
    with open(disk_path, "wb") as f:
        decompressed = hatanaka.decompress(req.content)
        # doesn't 404 I guess?
        if b"<!DOCTYPE html>" in decompressed:
            return None
        f.write(decompressed)
    return disk_path


def cors_get_station_lists_for_day(date: datetime) -> Iterable[str]:
    """
    Given a date, returns the stations that the US CORS network
    reports as available
    """
    url = "https://geodesy.noaa.gov/corsdata/rinex/"
    resp = requests.get(url + date.strftime("%Y/%j/"))

    pat = '<a href="..../">([a-z0-9]{4})/</a>'
    return re.findall(pat, resp.text)


def fetch_rinex_for_station(
    dog: Optional[AstroDog], time: GPSTime, station_name: str, partial: bool = False
) -> Optional[str]:
    """
    Given a particular time and station, get the rinex obs file that
    corresponds to it

    Args:
        dog: laika AstroDog object or None
        time: laika GPSTime object for the time in question
        station_name: string of the station in question
            station names are CORS names or similar (eg: 'slac')
        partial: whether to fetch preliminary (hourly) data, if available

    Returns:
        the string containing the file path, or None
    """

    if dog is None:
        dog = AstroDog(cache_dir=conf.cache_dir)

    # handlers for specific networks
    handlers = {
        "Korea": _download_korean_station,
        "Japan": _download_japanese_station,
        "Mongolia": _download_mongolian_station,
    }

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
                rinex_obs_file = _download_misc_igs_station(
                    dog, time, station_name, partial=partial
                )
            else:
                return None
        except hatanaka.hatanaka.HatanakaException:
            # not gonna handle this ourselves, sadly
            return None

    else:
        rinex_obs_file = handlers[network](dog, time, station_name, partial=partial)

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
    rinex_obs_file = fetch_rinex_for_station(dog, time, station_name)
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


def from_xarray_sat(rinex: xarray.Dataset, start_date: GPSTime) -> types.Observations:
    """
    Convert the georinex xarray for a satellite to Observations

    Args:
        xarray: the georinex xarray thing
        start_date: time at which tick 0 occurred

    Returns:
        Observations for the satellite
    """
    # truncate to observations with data
    if "C1" not in rinex:
        return cast(types.Observations, numpy.zeros(0, dtype=DENSE_TYPE))
    rinex = rinex.dropna("time", how="all", subset=["C1"])
    outp = numpy.zeros(rinex.dims["time"], dtype=DENSE_TYPE)

    obs_map = {"C1C": "C1", "C2C": "C2", "C2P": "P2", "L1C": "L1", "L2C": "L2"}
    for obs in ["C1C", "C2C", "L1C", "L2C"]:
        # if the channel doesn't exist, set to NaN
        if obs_map[obs] not in rinex:
            outp[obs][:] = numpy.nan
        else:
            outp[obs][:] = rinex[obs_map[obs]]

    # if the C2C channel is empty/crap, replace it with C2P
    if numpy.all(numpy.isnan(outp["C2C"])):
        outp["C2C"][:] = rinex[obs_map["C2P"]]

    timedeltas = rinex["time"].astype(numpy.datetime64).to_numpy() - numpy.datetime64(
        start_date.as_datetime()
    )
    outp["tick"] = (timedeltas / numpy.timedelta64(util.DATA_RATE, "s")).astype(int)
    return cast(types.Observations, outp)


def from_xarray(rinex: xarray.Dataset, start_date: GPSTime) -> types.DenseMeasurements:
    """
    Convert georinex's xarray format into our sparser format

    Args:
        rinex: the georinex xarray file
        start_date: when tick 0 occurred

    Returns:
        dense raw gps data
    """
    sv_dict_out = cast(types.DenseMeasurements, {})
    for svid in rinex.sv.to_numpy():
        sv_dict_out[svid] = from_xarray_sat(rinex.sel(sv=svid), start_date)
    return sv_dict_out


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
        station_name: the station for which we want data
        start_date: when index 0 occurred

    Returns:
        dense raw gps data

    Raises:
        DownloadError if the data could not be fetched

    TODO: caching of the results on disk? or should that happen later?
    """
    rinex_obs_file = fetch_rinex_for_station(dog, time, station_name)
    if rinex_obs_file is None:
        raise DownloadError

    rinex = georinex.load(rinex_obs_file, interval=30)
    return from_xarray(rinex, start_date)


def get_sat_info_old_okay(
    dog: AstroDog, start_time: GPSTime
) -> Dict[str, Tuple[numpy.ndarray, numpy.ndarray, float, float]]:
    """
    Wrapper around dog.get_all_sat_info that will use out-of-date data for
    GLONASS if we can't find any. GLONASS updates stuff real slow?

    Args:
        dog: AstroDog to use
        start_time: time for which we want data

    Returns:
        dict of PRNs to (position, velocity, offset1, offset2)
        same format as dog.get_all_sat_info
    """
    res = dog.get_all_sat_info(start_time)
    # missing GLONASS data
    if "R01" not in res:
        # and looking at something < 2 days old
        if GPSTime.from_datetime(datetime.utcnow()) - start_time < SECS_IN_DAY * 2:
            for i in range(1, 4):
                # most recent first, up to 3 days old
                eph = dog.get_nav("R01", start_time - SECS_IN_DAY * i)
                if eph:
                    break
            else:
                # can't get GLONASS data, whatever
                return res

            for prn, ephs in dog.nav.items():
                if not prn.startswith("R"):
                    continue
                res[prn] = ephs[-1].get_sat_info(start_time)
    return res


def populate_sat_info(
    dog: AstroDog,
    start_time: GPSTime,
    duration: timedelta,
    station_dict: types.StationPrnMap[types.Observations],
) -> None:
    """
    Populate the satellite locations for our measurements

    Args:
        dog: laika AstroDog to use
        start_time: when the 0th tick occurs
        duration: how long until the last tick
        station_dict: mapping to the Observations that need correcting

    TODO: can numba (or something) help us parallelize the lower loops?
    """

    satellites = {
        sat: idx for idx, sat in enumerate(get_sat_info_old_okay(dog, start_time))
    }
    tick_count = int(duration.total_seconds() / util.DATA_RATE)
    # get an accurate view of the satellites at 30 second intervals
    sat_info = numpy.zeros(
        (len(satellites), tick_count + 1), dtype=[("pos", "3f8"), ("vel", "3f8")]
    )

    for tick in range(tick_count + 1):
        tick_info = get_sat_info_old_okay(dog, start_time + util.DATA_RATE * tick)
        for svid, info in tick_info.items():
            if svid not in satellites:
                continue
            sat_info[satellites[svid]][tick] = (info[0], info[1])

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
        print("bad", station, sat)
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
            combined[prn] = cast(
                types.Observations, numpy.append(data1[prn], data2[prn])
            )

    return cast(types.DenseMeasurements, combined)


def populate_data(
    stations: Iterable[str],
    start_date: GPSTime,
    duration: timedelta,
    dog: AstroDog,
) -> Tuple[Dict[str, types.ECEF_XYZ], types.StationPrnMap[types.Observations]]:
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
    station_data = cast(types.StationPrnMap[types.Observations], {})

    for station in stations:
        gps_date = start_date
        while (gps_date) < start_date + duration.total_seconds():
            try:
                latest_data = data_for_station(
                    dog, gps_date, station, start_date=start_date
                )

                if station not in station_locs:
                    station_locs[station] = location_for_station(dog, gps_date, station)

            except DownloadError:
                continue
            except IndexError:
                print("index error: ", station)
                continue
            finally:
                gps_date += (1 * util.DAYS).total_seconds()
            if station not in station_data:
                station_data[station] = latest_data
            else:
                # we've already got some data, so merge it together
                # give mypy a hint here about our type aliases
                station_data[station] = merge_data(
                    cast(types.DenseMeasurements, station_data[station]),
                    latest_data,
                )

        # didn't download data, ignore it
        if station not in station_data:
            continue

    populate_sat_info(dog, start_date, duration, station_data)

    return station_locs, station_data


def download_and_process(
    argtuple: Tuple[GPSTime, str, bool]
) -> Tuple[GPSTime, str, Optional[str]]:
    """
    Fetch the data for a station at a date, return a path to the NetCDF4 version of it

    Args:
        argtuple: the date and station for which we want the data, and whether to get partial data

    Returns:
        date requested, station requested, and the path to the nc file, or
        None if it can't be retrieved
    """
    date, station, partial = argtuple

    if partial:
        char_code = char_code_for_partial(date)
    else:
        char_code = "0"

    # first search for already processed NetCDF4 files
    path_name = date.as_datetime().strftime(f"%Y/%j/{station}%j{char_code}.%yo.nc")
    for cache_folder in [
        "misc_igs_obs",
        "japanese_obs",
        "korean_obs",
        "mongolian_obs",
        "cors_obs",
    ]:
        fname = f"{conf.cache_dir}/{cache_folder}/{path_name}"
        if os.path.exists(fname):
            return date, station, fname

    rinex_obs_file = fetch_rinex_for_station(None, date, station, partial=partial)
    if rinex_obs_file is not None:
        if os.path.exists(rinex_obs_file + ".nc"):
            return date, station, rinex_obs_file + ".nc"
        rinex = georinex.load(rinex_obs_file, interval=30, use=["G", "R"], fast=False)
        rinex["time"] = rinex.time.astype(numpy.datetime64)
        rinex.to_netcdf(rinex_obs_file + ".nc")
        return date, station, rinex_obs_file + ".nc"
    return date, station, None


def parallel_populate_data(
    stations: Iterable[str],
    start_date: GPSTime,
    duration: timedelta,
    dog: AstroDog,
) -> Tuple[Dict[str, types.ECEF_XYZ], types.StationPrnMap[types.Observations]]:
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
    station_data = cast(types.StationPrnMap[types.Observations], {})

    # if we want < 1 day of data, get preliminary stuff
    partial = duration.days < 1

    to_download = []
    for station in stations:
        gps_date = start_date
        while gps_date < start_date + duration.total_seconds():
            to_download.append((gps_date, station, partial))
            if partial:
                gps_date += (1 * util.HOURS).total_seconds()
            else:
                gps_date += (1 * util.DAYS).total_seconds()

    with multiprocessing.Pool(DOWNLOAD_WORKERS) as pool:
        download_res = pool.map(download_and_process, to_download)

    downloaded_map = {
        # break it up like this to deal with GPSTime not being hashable
        (start_date.week, start_date.tow, station): result
        for start_date, station, result in download_res
    }

    for station in stations:
        gps_date = start_date
        while gps_date < start_date + duration.total_seconds():
            result = downloaded_map.get((gps_date.week, gps_date.tow, station))
            if partial:
                gps_date += (1 * util.HOURS).total_seconds()
            else:
                gps_date += (1 * util.DAYS).total_seconds()

            if result is None:
                continue

            latest_data = xarray.load_dataset(result)
            if station not in station_locs:
                station_locs[station] = latest_data.position

            dense_data = from_xarray(latest_data, start_date)
            if station not in station_data:
                station_data[station] = dense_data
            else:
                # we've already got some data, so merge it together
                # give mypy a hint here about our type aliases
                station_data[station] = merge_data(
                    cast(types.DenseMeasurements, station_data[station]),
                    dense_data,
                )

        # didn't download data, ignore it
        if station not in station_data:
            continue

    populate_sat_info(dog, start_date, duration, station_data)

    return station_locs, station_data
