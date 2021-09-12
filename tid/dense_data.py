"""
By default, Laika's objects are not very memory efficient
This file has wrappers/helpers/definitions for a more
space efficient format based on numpy
"""
from datetime import timedelta
from typing import cast, Dict, Iterable, List, Sequence, Tuple
import numpy

import georinex

from laika import AstroDog
from laika.gps_time import GPSTime
from laika.raw_gnss import GNSSMeasurement

from tid import get_data, tec, types, util

# dict of prn -> numpy.array(dtype=DENSE_TYPE)
DenseMeasurements = Dict[str, types.DenseDataType]

DENSE_TYPE = [
    ("tick", "i4"),  # tick number the observation was made
    ("C1C", "f8"),  # GNSS measurements, if available
    ("C2C", "f8"),
    ("L1C", "f8"),
    ("L2C", "f8"),
    ("sat_pos", "3f8"),  # satellite position XYZ ECEF in meters
]


def from_xarray(xarray, tick_offset: int = 0) -> DenseMeasurements:
    """ """
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

    # discard the empty slots (places where C1C is nan)
    good_ticks = numpy.where(numpy.logical_not(numpy.isnan(outp["C1C"])))[0]
    outp = outp[good_ticks]
    outp["tick"] = good_ticks + tick_offset
    return outp


def dense_data_for_station(
    dog: AstroDog, time: GPSTime, station_name: str, tick_offset: int = 0
) -> DenseMeasurements:
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
    rinex_obs_file = get_data.rinex_file_for_station(dog, time, station_name)
    if rinex_obs_file is None:
        raise get_data.DownloadError

    rinex = georinex.load(rinex_obs_file, interval=30)

    sv_dict_out: Dict[str, types.DenseDataType] = {}
    for sv in rinex.sv.to_numpy():
        sv_dict_out[sv] = from_xarray(rinex.sel(sv=sv), tick_offset=tick_offset)
    return sv_dict_out


def populate_sat_info(
    dog: AstroDog,
    start_time: GPSTime,
    duration: timedelta,
    station_dict: types.StationPrnMap[DenseMeasurements],
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


def merge_data(data1: DenseMeasurements, data2: DenseMeasurements) -> DenseMeasurements:
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


def populate(dog, date, duration, stations):
    station_table = {}
    for sta in stations:
        station_table[sta] = dense_data_for_station(dog, date, sta)


def populate_data(
    stations: Iterable[str],
    start_date,
    duration,
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
                latest_data = dense_data_for_station(
                    dog, gps_date, station, tick_offset=tick
                )
            except get_data.DownloadError:
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
            station_locs[station] = get_data.location_for_station(
                dog, gps_date, station
            )

    populate_sat_info(dog, start_date, duration, station_data)

    return station_locs, station_data
