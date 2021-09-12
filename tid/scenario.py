"""
Workhorse class/definitions. Scenario is the set of receivers for a given time.

This provides classes and related functions for the Scenario
"""
from __future__ import annotations  # defer type annotations due to circular stuff

from datetime import datetime, timedelta
from functools import lru_cache
from typing import cast, Dict, Iterable, Optional, Sequence, Tuple, Union
from pathlib import Path
import hashlib

import ruptures
import numpy
import h5py

from laika import AstroDog, constants
from laika.gps_time import GPSTime
from laika.lib import coordinates
from laika.rinex_file import DownloadError

from tid.config import Configuration
from tid.connections import Connection, ConnTickMap
from tid import bias_solve, dense_data, get_data, tec, types, util

from tid.util import get_dates_in_range as _get_dates_in_range

# load configuration data
conf = Configuration()

MIN_CON_LENGTH = 20  # 10 minutes worth of connection
DISCON_TIME = 4  # cycle slip for >= 4 samples without info
EL_CUTOFF = 0.25  # elevation cutoff in radians, shallower than this ignored


def _correct_satellite_info(dog, prn, data: types.DenseDataType) -> None:
    """
    Go through the observables data and do small satellite correction fixing stuff

    Args:
        data: the observables data to correct for one satellite
    """
    # TODO handle (rare) wrap-around case where correction pushes us back a week!
    adj_sec = data["recv_time_sec"] - data["C1C"] / constants.SPEED_OF_LIGHT
    for i, entry in enumerate(data):

        time_of_interest = GPSTime(week=int(entry["recv_time_week"]), tow=adj_sec[i])
        sat_info = dog.get_sat_info(prn, time_of_interest)
        # laika doesn't fall back to less-accurate NAV data if SP3 data is unavailable
        # if sat_info is empty, we can try it ourselves
        if sat_info is None:
            eph = dog.get_nav(prn, time_of_interest)
            if eph is None:
                continue
            sat_info = eph.get_sat_info(time_of_interest)

        # if it's still empty, it's a lost cause
        if sat_info is None:
            continue
        entry["sat_clock_err"] = sat_info[2]
        entry["sat_pos"] = sat_info[0]
        entry["sat_vel"] = sat_info[1]
        entry["is_processed"] = True


def _populate_data(
    stations: Iterable[str],
    date_list: Sequence[datetime],
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

    gps_start = GPSTime.from_datetime(date_list[0])

    for station in stations:
        for date in date_list:
            gps_date = GPSTime.from_datetime(date)
            try:
                latest_data = dense_data.dense_data_for_station(dog, gps_date, station)
            except DownloadError:
                continue
            if station not in station_data:
                station_data[station] = latest_data
            else:
                # we've already got some data, so merge it together
                station_data[station] = dense_data.merge_data(
                    station_data[station], latest_data
                )

        # didn't download data, ignore it
        if station not in station_data:
            continue

        if station not in station_locs:
            station_locs[station] = get_data.location_for_station(
                dog, gps_date, station
            )

        for prn, obs_data in station_data[station].items():
            _correct_satellite_info(dog, prn, obs_data)
            obs_data["tick"] = (
                (obs_data["recv_time_sec"] - gps_start.tow)
                + (obs_data["recv_time_week"] - gps_start.week)
                * gps_start.seconds_in_week
            ) // 30

    return station_locs, station_data


class Scenario:
    """
    This class stores information which is shared during each "engagement"

    So for a given set of parameters (date, stations, satellites, settings)
    this will store and process common items
    """

    def __init__(
        self,
        start_date: datetime,
        duration: timedelta,
        station_locs: Dict[str, types.ECEF_XYZ],
        station_data: types.StationPrnMap[types.DenseDataType],
        dog: AstroDog,
        conn_map=None,
    ) -> None:
        """
        Internal / raw constructor, probably not for use by mortals

        Args:
            start_date: start of the scenario

        """
        self.dog = dog
        self.cache_dir = dog.cache_dir
        self.start_date = start_date
        self.date_list = _get_dates_in_range(start_date, duration)
        self.duration = duration

        self.station_locs = station_locs
        self.station_data = station_data

        self.conn_map: Optional[types.StationPrnMap[ConnTickMap]] = conn_map

        # biases to be calculated of prn or station to clock bias in meters (C*time)
        # by some weird convention I don't get, these have opposite signs applied to them
        self.bias_solver: Optional[bias_solve.BiasSolver] = None
        self.sat_biases: Dict[str, float] = {}
        self.rcvr_biases: Dict[str, Tuple[float, float, float]] = {}

    def to_hdf5(self, fname: Path, *, overwrite=False) -> None:
        """
        Serialize/cache the scenario to hdf5 datastructure

        Args:
            fname: the path to which the data should be saved
            overwrite: whether the file should be overwritten if it already exists
        """
        mode = "w" if overwrite else "w-"
        with h5py.File(fname, mode) as fout:
            for station, sats in self.station_data.items():
                for prn, data in sats.items():
                    fout.create_dataset(f"data/{station}/{prn}", data=data)
            for station, loc in self.station_locs.items():
                fout[f"loc/{station}"] = loc
            fout.attrs.update(
                {
                    "start_date": self.start_date.timestamp(),
                    "duration": self.duration.total_seconds(),
                }
            )

    @classmethod
    def from_hdf5(cls, fname: Path, *, dog: Optional[AstroDog] = None) -> Scenario:
        """
        Deserialize/fetch the scenario from an hdf5 save file

        Args:
            fname: the path from which the data should be restored
            dog: a Laika AstroDog object (if unspecified will make a new one)
                that the scenario should use

        Returns:
            the cached scenario
        """
        if dog is None:
            dog = AstroDog(cache_dir=conf.cache_dir)
        with h5py.File(fname, "r") as fin:
            start_date = datetime.fromtimestamp(fin.attrs["start_date"])
            duration = timedelta(seconds=int(fin.attrs["duration"]))
            station_data = {
                station: {sat: ds[:] for sat, ds in group.items()}
                for station, group in fin["data"].items()
            }
            station_locs = {station: ds[:] for station, ds in fin["loc"].items()}

        return cls(
            start_date,
            duration,
            station_locs,
            cast(types.StationPrnMap[types.DenseDataType], station_data),
            dog,
        )

    @classmethod
    def from_daterange(
        cls,
        start_date: datetime,
        duration: timedelta,
        stations: Iterable[str],
        dog: Optional[AstroDog] = None,
        *,
        use_cache: bool = True,
    ) -> Scenario:
        """
        Args:
            start_date: when to start the scenario
            duration: how long the scenario should last
            stations: list of stations to use
            dog: Optional, AstroDog instance to use to manage data access
            use_cache: Optional, if should consider TID's cache

        Returns:
            scenario: The requested Scenario

        TODO: populate stations automatically?
        """

        if dog is None:
            dog = AstroDog(cache_dir=conf.cache_dir)
        cache_key = cls.compute_cache_key(start_date, duration, stations)
        cache_path = Path(conf.cache_dir) / "scenarios" / f"{cache_key}.hdf5"
        if use_cache and cache_path.exists():
            return cls.from_hdf5(cache_path, dog=dog)

        # date_list = _get_dates_in_range(start_date, duration)
        stations = set(stations)
        locs, data = dense_data.populate_data(
            stations, GPSTime.from_datetime(start_date), duration, dog
        )
        # locs, data = _populate_data(stations, date_list, dog)
        scn = cls(start_date, duration, locs, data, dog)

        if use_cache:
            cache_path.parent.mkdir(exist_ok=True)
            scn.to_hdf5(cache_path, overwrite=True)
        return scn

    @staticmethod
    def compute_cache_key(
        start_date: datetime, duration: timedelta, stations: Iterable[str]
    ) -> str:
        """
        Given scenario arguments, calculate a unique id for those arguments

        Args:
            start_date: when the scenario starts
            duration: how long the scenario lasts
            stations: the name of the stations to use

        Returns:
            unique string for the given arguments
        """
        hasher = hashlib.md5()
        hasher.update(repr(sorted(stations)).encode())
        hasher.update(start_date.isoformat().encode())
        hasher.update(repr(duration.total_seconds()).encode())
        return hasher.hexdigest()

    @lru_cache(maxsize=None)
    def _station_converter(self, station: str) -> coordinates.LocalCoord:
        """
        Cached version of Laika's local coordinate transform for a station.

        Args:
            station: the station name

        Returns:
            the coordinate transform for that station
        """
        return coordinates.LocalCoord.from_ecef(self.station_locs[station])

    def station_el(
        self, station: str, sat_pos: Union[types.ECEF_XYZ, types.ECEF_XYZ_LIST]
    ) -> Union[types.ECEF_XYZ, types.ECEF_XYZ_LIST]:
        """
        Helper to get elevations of satellite looks more efficiently.
        This re-uses the station converters which helps performance

        Args:
            station: station name
            sat_pos: numpy array of XYZ ECEF satellite positions in meters
                must have shape (?, 3)

        Returns:
            elevation in radians (will have same length as sat_pos)
        """
        sat_ned = self._station_converter(station).ecef2ned(sat_pos)
        sat_range = numpy.linalg.norm(sat_ned, axis=1)
        return numpy.arcsin(-sat_ned[..., 2] / sat_range)

    def get_extent(self) -> Tuple[float, float, float, float]:
        """
        Get a rough idea of the geographic region we are working with.

        Returns:
            min longitude, max longitude, min latitude, max latitude
            all in degrees
        """
        max_lat, max_lon = -numpy.inf, -numpy.inf
        min_lat, min_lon = numpy.inf, numpy.inf
        for loc in self.station_locs.values():
            lat, lon, _ = coordinates.ecef2geodetic(loc)
            max_lat = max(max_lat, lat)
            max_lon = max(max_lon, lon)
            min_lat = min(min_lat, lat)
            min_lon = min(min_lon, lon)

        # add a degree of padding around the edge
        return min_lon - 1, max_lon + 1, min_lat - 1, max_lat + 1

    def get_filtered_vtec_data(
        self,
    ) -> Tuple[
        types.StationPrnMap[Sequence[float]],
        types.StationPrnMap[Sequence[Optional[Tuple[float, float]]]],
    ]:
        """
        Get organized vtec data for this scenario.

        Returns:
            map of station -> prn -> filtered vtec data, one per tick
            map of station -> prn -> (lat, lon values or None if no data), one per tick
        """
        vtecs = cast(types.StationPrnMap[Sequence[float]], {})
        ipps = cast(types.StationPrnMap[Sequence[Optional[Tuple[float, float]]]], {})
        for station in self.conn_map.keys():
            for prn in self.conn_map[station].keys():
                if not self.conn_map[station][prn].connections:
                    continue
                if station not in vtecs:
                    vtecs[station] = {}
                if station not in ipps:
                    ipps[station] = {}
                vtecs[station][prn] = self.conn_map[station][prn].get_filtered_vtecs()
                ipps[station][prn] = self.conn_map[station][prn].get_ipps_latlon()
        return vtecs, ipps

    def export_vtec_data(self, fname: Path) -> None:
        """
        Write out a big matrix with filtered vtec data to easily share it around

        Args:
            fname: path for where the data should be written
        """
        tick_count = int(self.duration.total_seconds() / util.DATA_RATE)
        stations = sorted(self.station_data.keys())
        sats = sorted(
            set(
                sum(
                    [
                        list(station_map.keys())
                        for station_map in self.conn_map.values()
                    ],
                    start=[],
                )
            )
        )
        max_obs = len(stations) * len(sats)

        res = numpy.zeros(
            (tick_count, max_obs), dtype=[("vtec", "f8"), ("latlon", "2f8")]
        )

        for station in self.conn_map.keys():
            station_idx = stations.index(station)
            for prn in self.conn_map[station].keys():
                sat_idx = sats.index(prn)
                if not self.conn_map[station][prn].connections:
                    continue
                vtecs = self.conn_map[station][prn].get_filtered_vtecs()
                ipps = self.conn_map[station][prn].get_ipps_latlon()
                for tick in range(tick_count):
                    latlon = ipps[tick]
                    if latlon is None:
                        continue
                    res[tick][station_idx * len(sats) + sat_idx] = (vtecs[tick], latlon)
        with h5py.File(fname, "w") as fout:
            fout.create_dataset("data", data=res, compression="gzip")

    def get_glonass_chan(
        self, prn: str, observations: types.DenseDataType
    ) -> Optional[int]:
        """
        Get the GLONASS channel for this satellite for these observations


        Args:
            prn: the prn of interest
            observations: observations from which to get the time info

        Returns:
            the integer channel on which this satellite is operating, or none if it
            could not be found
        """
        time = GPSTime.from_datetime(self.start_date) + util.DATA_RATE * int(
            observations[0]["tick"]
        )
        return self.dog.get_glonass_channel(prn, time)

    def get_frequencies(
        self, prn: str, observations: types.DenseDataType
    ) -> Optional[Tuple[float, float]]:
        """
        Get the channel 1 and 2 frequencies corresponding to the given observations

        Args:
            prn: the prn of interest
            observations: observations from which to get the time info

        Returns:
            the channel 1 and 2 frequencies in Hz, or None if they could not be found
        """
        time = GPSTime.from_datetime(self.start_date) + util.DATA_RATE * int(
            observations[0]["tick"]
        )
        f1, f2 = [self.dog.get_frequency(prn, time, band) for band in ("C1C", "C2C")]
        # if the lookup didn't work, we can't proceed
        if not f1 or not f2:
            return None
        return f1, f2

    def _get_connections_internal(
        self,
        station,
        prn,
        observations: types.DenseDataType,
        el_cutoff: float = EL_CUTOFF,
    ) -> Iterable[Connection]:
        """
        Get a list of Connections given observations

        Args:
            observations: the observations to use
            el_cutoff: ignore signals below this number in radians

        Returns:
            a list of Connection objects
        """
        if len(observations) < MIN_CON_LENGTH:
            return []

        bkpoints = set()

        # first pass: when tickcount jumps by >= DISCON_TIME
        bkpoints |= set(numpy.where(numpy.diff(observations["tick"]) >= DISCON_TIME)[0])

        mw_signal = tec.melbourne_wubbena(
            self.get_frequencies(prn, observations), observations
        )
        # if this calculation failed, we don't have proper dual channel info anyway
        if mw_signal is None:
            return []

        # second pass: when mw_signal value is NaN
        bkpoints |= set(numpy.where(numpy.isnan(mw_signal))[0])

        # third pass: elevation cutoff
        bkpoints |= set(
            numpy.where(self.station_el(station, observations["sat_pos"]) < el_cutoff)[
                0
            ]
        )

        # final pass: run ruptures on the remaining mw_signal contiguous chunks
        binseg = ruptures.Binseg(model="l2")
        bkpoint_list = sorted(bkpoints)
        ruptures_bkpoints = set()
        for i, bkpoint in enumerate(bkpoint_list):
            if i == 0:  # first breakpoint
                start = 0
            else:
                start = bkpoint_list[i - 1]
            count = bkpoint - start
            if count < MIN_CON_LENGTH:
                continue
            bkpts = binseg.fit_predict(
                mw_signal[start:bkpoint], pen=count / numpy.log(count)
            )
            ruptures_bkpoints |= set(bkpts)

        if len(bkpoint_list) > 0:
            # and one for the last section
            start = bkpoint_list[-1]
            bkpoint = len(observations) - 1
            count = bkpoint - start
            if count >= MIN_CON_LENGTH:
                bkpts = binseg.fit_predict(
                    mw_signal[start:bkpoint], pen=count / numpy.log(count)
                )
                ruptures_bkpoints |= set(bkpts)

        # include everything EXCEPT for these points
        # and separate chunks by these points
        # don't include segments that are too short
        partition_points = sorted(ruptures_bkpoints | set(bkpoints))

        connections = []
        for i, bkpoint in enumerate(partition_points):
            if i == 0:
                start = 0
            else:
                # not inclusive--start one after
                start = partition_points[i - 1] + 1
            count = (bkpoint - 1) - start
            if count < MIN_CON_LENGTH:
                continue

            connections.append(
                Connection(
                    self,
                    station,
                    prn,
                    start,
                    bkpoint - 1,
                )
            )
        if len(partition_points) > 0:
            start = partition_points[-1] + 1
            bkpoint = len(observations) - 1
            count = bkpoint - start
            if count >= MIN_CON_LENGTH:
                connections.append(
                    Connection(
                        self,
                        station,
                        prn,
                        start,
                        bkpoint,
                    )
                )

        return connections

    def make_connections(self):
        """
        Generate and store our lists of connections
        """
        if self.conn_map:
            return

        self.conn_map = {}
        for station, svmap in self.station_data.items():
            self.conn_map[station] = {}
            for prn, observations in svmap.items():
                cons = self._get_connections_internal(station, prn, observations)
                for con in cons:
                    con.correct_ambiguities()
                self.conn_map[station][prn] = ConnTickMap(cons)

    def solve_biases(self):
        assert self.conn_map
        self.bias_solver = bias_solve.SimpleBiasSolver(self)
        self.sat_biases, self.rcvr_biases = self.bias_solver.solve_biases()
