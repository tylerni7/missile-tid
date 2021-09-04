"""
Workhorse class/definitions. Scenario is the set of receivers for a given time.

This provides classes and related functions for the Scenario
"""
from __future__ import annotations  # defer type annotations due to circular stuff

from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Iterable, Optional, Tuple

import ruptures
import numpy

from laika import AstroDog, constants
from laika.gps_time import GPSTime
from laika.lib import coordinates
from laika.rinex_file import DownloadError

from tid.config import Configuration
from tid.connections import Connection, ConnTickMap
from tid import dense_data, get_data, tec, util

# load configuration data
conf = Configuration()

MIN_CON_LENGTH = 20  # 10 minutes worth of connection
DISCON_TIME = 4  # cycle slip for >= 4 samples without info
EL_CUTOFF = 0.25  # elevation cutoff in radians, shallower than this ignored


def get_dates_in_range(start_date: datetime, duration: timedelta) -> Iterable[datetime]:
    """
    Get a list of dates, starting with start_date, each 1 day apart

    Args:
        start_date: the first date to include
        duration: how long to include

    Returns:
        list of dates, each separated by 1 day
    """
    dates = [start_date]
    last_date = start_date + timedelta(days=1)
    while last_date > dates[0] + duration:
        dates.append(last_date)
        last_date += timedelta(days=1)
    return dates


class Scenario:
    """
    This class stores information which is shared during each "engagement"

    So for a given set of parameters (date, stations, satellites, settings)
    this will store and process common items
    """

    def __init__(
        self, start_date: datetime, duration: timedelta, stations: Iterable[str]
    ) -> None:
        """
        Args:
            start_date: when to start the scenario
            duration: how long the scenario should last
            stations: list of stations to use

        TODO: populate stations automatically?
        """
        self.dog = AstroDog(cache_dir=conf.cache_dir)
        self.cache_dir: str = conf.cache_dir
        self.start_date = start_date
        self.duration = duration
        self.date_list = get_dates_in_range(start_date, duration)
        self.stations = set(stations)

        # dict of station names -> XYZ ECEF locations in meters
        self.station_locs: Dict[str, Iterable[float]] = {}
        # dict of station names -> dict of prn -> numpy observation data
        self.station_data: Dict[str, Dict[str, numpy.array]] = {}

        self.conn_map = None

        # biases to be calculated of prn or station to clock bias in meters (C*time)
        # by some weird convention I don't get, these have opposite signs applied to them
        self.sat_biases: Dict[str, float] = {}
        self.rcvr_biases: Dict[str, float] = {}

        self._populate_data()

    def _populate_data(self) -> None:
        """
        Download/populate the station data and station location info

        TODO: is this a good place to be caching results?
        """
        assert len(self.station_data) == 0, "data already populated"

        for station in self.stations:
            for date in self.date_list:
                gps_date = GPSTime.from_datetime(date)
                try:
                    latest_data = dense_data.dense_data_for_station(
                        self.dog, gps_date, station
                    )
                except DownloadError:
                    continue
                if station not in self.station_data:
                    self.station_data[station] = latest_data
                else:
                    # we've already got some data, so merge it together
                    self.station_data[station] = dense_data.merge_data(
                        self.station_data[station], latest_data
                    )
                if station not in self.station_locs:
                    self.station_locs[station] = get_data.location_for_station(
                        self.dog, gps_date, station
                    )

            for _, obs_data in self.station_data[station].items():
                self.correct_satellite_info(obs_data)

    def correct_satellite_info(self, data: numpy.array) -> None:
        """
        Go through the observables data and do small satellite correction fixing stuff

        Args:
            data: the observables data to correct for one satellite
        """
        adj_sec = data["recv_time_sec"] - data["C1C"] / constants.SPEED_OF_LIGHT
        for i, entry in enumerate(data):
            time_of_interest = GPSTime(
                week=int(entry["recv_time_week"]), tow=adj_sec[i]
            )
            sat_info = self.dog.get_sat_info(entry["prn"], time_of_interest)
            # laika doesn't fall back to less-accurate NAV data if SP3 data is unavailable
            # if sat_info is empty, we can try it ourselves
            if sat_info is None:
                eph = self.dog.get_nav(entry["prn"], time_of_interest)
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

    def station_el(self, station, sat_pos) -> float:
        """
        Helper to get elevations of satellite looks more efficiently.
        This re-uses the station converters which helps performance

        Args:
            station: station name
            sat_pos: the XYZ ECEF satellite position in meters

        Returns:
            elevation in radians
        """
        sat_ned = self._station_converter(station).ecef2ned(sat_pos)
        sat_range = numpy.linalg.norm(sat_ned, axis=1)
        return numpy.arcsin(-sat_ned[..., 2] / sat_range)

    def get_vtec_data(self):
        pass

    def get_frequencies(
        self, observations: numpy.array
    ) -> Optional[Tuple[float, float]]:
        """
        Get the channel 1 and 2 frequencies corresponding to the given observations

        Args:
            observations: observations containing time and PRN information
        """
        prn = observations[0]["prn"]
        time = GPSTime(
            week=int(observations[0]["recv_time_week"]),
            tow=observations[0]["recv_time_sec"],
        )
        f1, f2 = [self.dog.get_frequency(prn, time, band) for band in ("C1C", "C2C")]
        # if the lookup didn't work, we can't proceed
        if not f1 or not f2:
            return None
        return f1, f2

    def _get_connections_internal(
        self, observations: numpy.array, el_cutoff: float = EL_CUTOFF
    ) -> Iterable[Connection]:
        """
        Get a list of Connections given observations

        Args:
            observations: the observations to use
            el_cutoff: ignore signals below this number in radians

        Returns:
            a list of Connection objects
        """
        bkpoints = set()
        station = observations[0]["station"]
        prn = observations[0]["prn"]

        # first pass: when tickcount jumps by >= DISCON_TIME
        bkpoints |= set(numpy.where(numpy.diff(observations["tick"]) >= DISCON_TIME)[0])

        mw_signal = tec.melbourne_wubbena(self, observations)
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

            connections.append(Connection(self, station, prn, start, bkpoint - 1))
        start = partition_points[-1] + 1
        bkpoint = len(observations) - 1
        count = bkpoint - start
        if count >= MIN_CON_LENGTH:
            connections.append(Connection(self, station, prn, start, bkpoint))

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
                cons = self._get_connections_internal(observations)
                for con in cons:
                    con.correct_ambiguities()
                self.conn_map[station][prn] = ConnTickMap(cons)