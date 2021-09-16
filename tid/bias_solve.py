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
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy
from scipy import optimize, sparse

from laika.lib import coordinates

from tid.util import DATA_RATE


# deal with circular type definitions for Scenario
if TYPE_CHECKING:
    from tid.scenario import Scenario
    from tid.connections import Connection

LAT_RES = 2.5  # 2.5 degrees
LON_RES = 5  # 5 degrees
TIME_RES = 60 * 15  # 15 minutes

TEC_GUESS = 25  # TEC estimate, as it isn't centered about 0


class BiasSolver(ABC):
    """
    Generic Bias solver Abstract class. All Bias solvers must inherit from this
    """

    @abstractmethod
    def __init__(self, scenario: Scenario) -> None:
        """Create a bias solver"""

    @abstractmethod
    def solve_biases(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float, float]]]:
        """Solve the biases, return sat and station biases"""


def _sparse_lsq_solve(
    matrix_a_list: numpy.ndarray,
    matrix_b: numpy.ndarray,
) -> numpy.ndarray:
    """
    Solve the least squares optimization problem for a sparse design matrix

    Args:
        matrix_a_list: the design matrix, in dictionary-of-keys format to be expanded
        matrix_b: the target values

    Returns:
        the optimum values for the least squares problem posed
    """
    # construct the full A matrix
    matrix_a = sparse.csr_matrix(
        (
            matrix_a_list["value"],
            (matrix_a_list["row"], matrix_a_list["col"]),
        ),
    )

    return optimize.lsq_linear(matrix_a, matrix_b).x


EntryVector = Tuple[
    float, Tuple[float, float, float], float, int, int, float, Optional[int]
]


class SimpleBiasSolver(BiasSolver):
    """
    This bias solved just looks for points which are coincidental.
    Nothing too fancy, and should be replaced with Thin Spline Model from mgnute
    """

    def __init__(self, scenario: Scenario) -> None:
        # scenario object, from which we will extract some useful info
        self.scenario = scenario

        self.stations = sorted(self.scenario.conn_map.keys())
        self.sats = sorted(list(self._get_sats()))

        self.total_tec_values = 0

    def _get_sats(self) -> Iterable[str]:
        """
        Return all the PRNs seen by all stations across all observations

        Returns:
            a set of all PRNs
        """
        sats = set()
        for station_dict in self.scenario.conn_map.values():
            sats |= set(station_dict.keys())
        return sats

    def _add_connection(
        self, connection: Connection, entries: List[EntryVector]
    ) -> None:
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

        # we can average out data and prevent extra entries by stashing stuff for the same
        # tec measurement together
        holding_cell: Dict[Tuple[float, float, float], List[Tuple[float, float]]] = {}
        for tick, (lat, lon), (vtec, slant) in zip(
            rounded_ticks, rounded_lat_lons, connection.vtecs.T
        ):
            tec_loc = (lat, lon, tick)
            if tec_loc not in holding_cell:
                holding_cell[tec_loc] = []
            holding_cell[tec_loc].append((vtec, slant))

        sat_idx = self.sats.index(connection.prn)
        station_idx = self.stations.index(connection.station)
        # go back and coalesce stuff for each tic
        for tec_loc, hits in holding_cell.items():
            vtec_total = sum(hit[0] for hit in hits)
            slant_total = sum(hit[1] for hit in hits)
            entries.append(
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

    def _coalesce_entries(
        self, entries: Sequence[EntryVector]
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        In _add_connection, we used python lists temporarily to store data.
        Here we'll squish that all back into a temporary matrix

        Args:
            entries: the list of entry data created at in _add_connection

        Returns:
            numpy array of the target values to reach in least square optimization
            numpy array of the dictionary-of-keys entries for a sparse array implementation
                which will be the design matrix for our least squares optimization
        """
        # first find which tec_locs were used
        counts = Counter(entry[1] for entry in entries)
        tec_id_map = {}
        idx = 0
        for tec_loc, count in counts.items():
            if count > 1:
                tec_id_map[tec_loc] = idx
                idx += 1
        self.total_tec_values = len(tec_id_map)

        # this matrix represents the unknowns for all our observations
        # the format is something like rows of
        # [true vTEC values][prn errors][station errors 0th order, station errors 1st order]
        # for now represent it as a list of dictionaries-of-keys
        # then later we can convert this to a sparse.csr_matrix
        matrix_a_list = numpy.zeros(
            65536,  # guess a default size, not too harmful to get it wrong
            dtype=[
                ("row", numpy.int32),
                ("col", numpy.int32),
                ("value", numpy.float64),
            ],
        )
        matrix_a_size = 0  # number of entries in the matrix
        b_values = []

        def _mat_insert(row: int, col: int, value: float) -> None:
            """
            Helper function to insert data into a memory efficient format for array conversion

            Args:
                row: the row of the data entry
                col: the column of the data entry
                value: the value of the data entry

            Note:
                modifies matrix_a_list and matrix_a_size from outer scope
            """
            nonlocal matrix_a_list, matrix_a_size
            if matrix_a_size >= matrix_a_list.shape[0]:
                # double the size
                matrix_a_list = numpy.resize(matrix_a_list, matrix_a_size * 2)

            matrix_a_list[matrix_a_size] = (row, col, value)
            matrix_a_size += 1

        measurements = 0
        for entry in entries:
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
            b_values.append(vtec_total - hit_cnt * TEC_GUESS)
            _mat_insert(measurements, tec_idx, hit_cnt)
            _mat_insert(
                measurements,
                self.total_tec_values + sat_idx,
                -slant_total,
            )
            if glonass_chan is not None:
                # correction for GLONASS: offset + linear component
                _mat_insert(
                    measurements,
                    self.total_tec_values + len(self.sats) + station_idx * 3 + 1,
                    slant_total,
                )
                _mat_insert(
                    measurements,
                    self.total_tec_values + len(self.sats) + station_idx * 3 + 2,
                    slant_total * glonass_chan,
                )
            else:
                # correction for GPS: single entry
                _mat_insert(
                    measurements,
                    self.total_tec_values + len(self.sats) + station_idx * 3,
                    slant_total,
                )
                # explicitly put in the 0 value to make sure the matrix size is correct
                _mat_insert(
                    measurements,
                    self.total_tec_values + len(self.sats) + station_idx * 3 + 2,
                    0,
                )
            measurements += 1

        matrix_a_list = numpy.resize(matrix_a_list, matrix_a_size)
        return numpy.array(b_values), matrix_a_list

    def solve_biases(
        self,
    ) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float, float]]]:
        """
        Solve the satellite and station biases

        Returns:
            dictionary mapping satellite PRNs to their biases (in meters)
            dictionary mapping station names to their bias vectors (GPS, GLONASS_0, GLONASS_1)
        """

        entries: List[EntryVector] = []
        for prn_map in self.scenario.conn_map.values():
            for conn_tick_map in prn_map.values():
                for connection in conn_tick_map.connections:
                    self._add_connection(connection, entries)

        matrix_b, matrix_a_list = self._coalesce_entries(entries)

        res = _sparse_lsq_solve(matrix_a_list, matrix_b)

        sat_biases = dict(
            zip(
                self.sats,
                res[self.total_tec_values : self.total_tec_values + len(self.sats)],
            )
        )
        remaining = res[self.total_tec_values + len(self.sats) :]
        station_biases = dict(
            zip(self.stations, numpy.array(remaining).reshape((len(remaining) // 3, 3)))
        )
        return sat_biases, station_biases
