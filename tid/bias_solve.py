"""
GNSS connections have two main bias sources of interest to us:
station bias (clock offsets/antenna paths on the receiver)
satellite bias (clock offsets/antenna paths on the receiver)

At a high level, we need to examine VTEC measurements and
look for when independent measurements calculate the VTEC
value for the "same" place and time.

By assuming VTECs are constant for certain spatio-temporal
resolutions, and with enough data, we can then resolve
the biases.

Biases we want to solve:
    1 x number of GPS satellites (offset)
    1 x number of GLONASS satellites (offset)
    2 x number of stations (offset + frequency term for different GLONASS freqs)
"""
from __future__ import annotations  # defer type annotations due to circular stuff

from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import numpy
from scipy import optimize, sparse

from laika.lib import coordinates

from tid import connections
from tid.util import DATA_RATE


# deal with circular type definitions for Scenario
if TYPE_CHECKING:
    from tid.scenario import Scenario
    from tid.connections import Connection

LAT_RES = 2.5  # 1 degrees
LON_RES = 5  # 1 degrees
TIME_RES = 60 * 15  # 15 minutes


class BiasSolver(ABC):
    @abstractmethod
    def __init__(self, scenario: Scenario) -> None:
        """Create a bias solver"""

    @abstractmethod
    def solve_biases(self) -> None:
        """Solve the biases"""


class SimpleBiasSolver(BiasSolver):
    """
    This bias solved just looks for points which are coincidental.
    Nothing too fancy, and should be replaced with Thin Spline Model from mgnute
    """

    def __init__(self, scenario: Scenario) -> None:
        # scenario object, from which we will extract some useful info
        self.scenario = scenario

        self.stations = sorted(self.scenario.stations)
        self.sats = sorted(list(self._get_sats()))

        self.total_tec_values = 0

        # for LSQ: we will minimize Ax - b
        # this matrix represents the unknowns for all our observations
        # the format is something like rows of
        # [true vTEC values][prn errors][station errors 0th order, station errors 1st order]
        # for now represent it as a list of dictionaries-of-keys
        # then later we can convert this to a sparse.csr_matrix
        self.A_matrix_list = numpy.zeros(
            1048576,  # guess a default size, not too harmful to get it wrong though
            dtype=[("row", numpy.int32), ("col", numpy.int32), ("value", numpy.float)],
        )
        self.A_matrix_size = 0

        self.entries = []

        # list of measured values to use for our LSQ result
        self.b_values: List[float] = []
        self.measurements = 0

    def mat_insert(self, row: int, col: int, value: float) -> None:
        """
        Helper function to insert data into a memory efficient format for array conversion

        Args:
            row: the row of the data entry
            col: the column of the data entry
            value: the value of the data entry
        """
        if self.A_matrix_size >= self.A_matrix_list.shape[0]:
            # double the size
            self.A_matrix_list = numpy.resize(
                self.A_matrix_list, self.A_matrix_size * 2
            )

        self.A_matrix_list[self.A_matrix_size] = (row, col, value)
        self.A_matrix_size += 1

    def tec_col_idx(self, lat: float, lon: float, tick: int) -> int:
        """
        Given a rounded latitude and longitude and tick at which a measurement
        occured, what is the proper column in the A matrix for the TEC measurement?

        Args:
            latitude: latitude in degrees, rounded to LAT_RES
            longitude: longitude in degrees, rounded to LON_RES
            tick: tick number, rounded to TIME_RES seconds

        Returns:
            integer index for the corresponding column
        """
        ll_offset = (lat - self.smallest_lat) / LAT_RES * self.coord_width + (
            lon - self.smallest_lon
        )
        tick_offset = self.coord_block_size * (tick / (TIME_RES / DATA_RATE))
        assert int(tick_offset + ll_offset) > 0
        return int(tick_offset + ll_offset)

    def sat_bias_col_idx(self, prn: str) -> int:
        """
        Given a prn, what is the proper column in the A matrix for the corresponding
        bias value?

        Args:
            prn: prn name of the satellite

        Returns:
            integer index for corresponding column

        Note:
            this is just called once per connection, so not as timing sensitive as tec_col_idx
        """
        return self.total_tec_values + self.sats.index(prn)

    def rcvr_bias_col_idx(self, station: str) -> Tuple[int, Optional[int]]:
        """
        Given a station name, what is the proper column in the A matrix for the corresponding
        bias value for 0th and 1st order corrections

        Args:
            station: name of the station

        Returns:
            integer index for corresponding column for 0th order correction
            and a second index for the 1st order correction

        Note:
            this is just called once per connection, so not as timing sensitive as tec_col_idx
        """
        idx = self.total_tec_values + len(self.sats) + 2 * self.stations.index(station)
        return idx, idx + 1

    def _get_sats(self) -> Iterable[str]:
        """
        Return all the PRNs seen by all stations across all observations

        Returns:
            a set of all PRNs
        """
        sats = set()
        for station_dict in self.scenario.station_data.values():
            sats |= set(station_dict.keys())
        return sats

    def add_connection(self, connection: Connection) -> None:
        """
        Add the data from the given connection to the matrices for the LSQ bias calculation

        Args:
            connection: the connection whose data we wish to add
        """
        # round time (ticks) to the desired amount
        tick_scale_factor = TIME_RES / DATA_RATE
        rounded_ticks = (
            numpy.round(connection.ticks / tick_scale_factor, 0) * tick_scale_factor
        )

        # round lat and lon to the desired amount
        ll_scale_factor = numpy.array([LAT_RES, LON_RES])
        scaled_lat_lons = (
            coordinates.ecef2geodetic(connection.ipps)[..., 0:2] / ll_scale_factor
        )
        rounded_lat_lons = numpy.round(scaled_lat_lons, 0) * ll_scale_factor

        holding_cell = {}
        for tick, (lat, lon), (vtec, slant) in zip(
            rounded_ticks, rounded_lat_lons, connection.vtecs.T
        ):
            # stec = true_tec / slant + sat_bias + recv_bias + glonass_chan*recv_glonass_bias
            # vtec = true_tec + slant(sat_bias + recv_bias + glonass_chan*recv_glonass_bias)
            tec_loc = (lat, lon, tick)
            if tec_loc not in holding_cell:
                holding_cell[tec_loc] = []
            holding_cell[tec_loc].append((vtec, slant))
            # set up this row of the A matrix

        sat_idx = self.sats.index(connection.prn)
        station_idx = self.stations.index(connection.station)
        for tec_loc, hits in holding_cell.items():
            vtec_total = sum(hit[0] for hit in hits)
            slant_total = sum(hit[1] for hit in hits)
            self.entries.append(
                (
                    vtec_total,
                    tec_loc,
                    len(hits),
                    sat_idx,
                    station_idx,
                    slant_total,
                    connection.glonass_chan if connection.is_glonass else None,
                )
            )

    def coalesce_entries(self):
        # first find which tec_locs were used
        counts = Counter(entry[1] for entry in self.entries)
        tec_id_map = {}
        idx = 0
        for tec_loc, count in counts.items():
            if count > 1:
                tec_id_map[tec_loc] = idx
                idx += 1
        self.total_tec_values = len(tec_id_map)

        for entry in self.entries:
            (
                vtec_total,
                tec_loc,
                hit_cnt,
                sat_idx,
                station_idx,
                slant_total,
                glonass_chan,
            ) = entry
            tec_idx = tec_id_map.get(tec_loc)
            if tec_idx is None:
                continue
            self.b_values.append(vtec_total)
            self.mat_insert(self.measurements, tec_idx, hit_cnt)
            self.mat_insert(
                self.measurements, self.total_tec_values + sat_idx, -slant_total
            )
            if glonass_chan is not None:
                self.mat_insert(
                    self.measurements,
                    self.total_tec_values + len(self.sats) + station_idx * 3 + 1,
                    slant_total,
                )
                self.mat_insert(
                    self.measurements,
                    self.total_tec_values + len(self.sats) + station_idx * 3 + 2,
                    slant_total * glonass_chan,
                )
            else:
                self.mat_insert(
                    self.measurements,
                    self.total_tec_values + len(self.sats) + station_idx * 3,
                    slant_total,
                )
            self.measurements += 1

    def lsq_bias_solve(self):
        # construct the full A matrix
        self.A_matrix_list = numpy.resize(self.A_matrix_list, self.A_matrix_size)
        A_mat = sparse.csr_matrix(
            (
                self.A_matrix_list["value"],
                (self.A_matrix_list["row"], self.A_matrix_list["col"]),
            ),
        )
        b_mat = numpy.array(self.b_values[: self.measurements])

        res = optimize.lsq_linear(A_mat, b_mat)
        res.x
        return res

    def solve_biases(self):
        for prn_map in self.scenario.conn_map.values():
            for conn_tick_map in prn_map.values():
                for connection in conn_tick_map.connections:
                    self.add_connection(connection)
        self.coalesce_entries()
        res = self.lsq_bias_solve()

        sat_biases = dict(
            zip(
                self.sats,
                res.x[self.total_tec_values : self.total_tec_values + len(self.sats)],
            )
        )
        remaining = res.x[self.total_tec_values + len(self.sats) :]
        station_biases = dict(
            zip(self.stations, numpy.array(remaining).reshape((len(remaining) // 3, 3)))
        )
        return sat_biases, station_biases
